import re
from pathlib import Path
from scipy import sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForMaskedLM

df = pd.read_csv("data/processed/preprocessed_abstracts.csv")

# Task 3.2: Leak-free 70% / 15% / 15% split grouped by 'source_id'
gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss_temp.split(df, df["target"], groups=df["source_id"]))

train_df = df.iloc[train_idx].copy().reset_index(drop=True)
temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss_val_test.split(temp_df, temp_df["target"], groups=temp_df["source_id"]))

val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

assert set(train_df["source_id"]).isdisjoint(set(val_df["source_id"])), "Leakage between Train and Val!"
assert set(train_df["source_id"]).isdisjoint(set(test_df["source_id"])), "Leakage between Train and Test!"
assert set(val_df["source_id"]).isdisjoint(set(test_df["source_id"])), "Leakage between Val and Test!"

# Task 3.1: Extract Assigned Stylometric Features (#10, #32, #54, #76)
PASSIVE_REGEX = re.compile(
    r"(?:"
    r"\b(?:تم|يتم|تتم|جرى|يجري|تجري)\s+(?:[\u0621-\u064A]{3,})|"
    r"\b(?:يُ|تُ|نُ|أُ)[\u0621-\u064A]+|"
    r"\b[يتنأ]ُ[\u0621-\u064A]+|"
    r"\b(?:يعتبر|تعتبر|يعد|تعد|يلاحظ|تلاحظ|لوحظ|أجريت|اجريت|نوقشت|استخدمت|صممت|طورت|طبقت)\b"
    r")"
)

def extract_assigned_stylometric_features(text: str) -> pd.Series:
    raw = str(text)
    words = [w for w in raw.split() if len(w) > 0]
    f10_avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    f32_double_quotes = len(re.findall(r'[\"“”«»]', raw))
    f54_foreign_letters = len(re.findall(r"[A-Za-z]", raw))
    sentences = [s.strip() for s in re.split(r"[.؟!؛\n]+", raw) if s.strip()]
    f76_passive_sentences = sum(1 for s in sentences if PASSIVE_REGEX.search(s))

    return pd.Series(
        [f10_avg_word_len, f32_double_quotes, f54_foreign_letters, f76_passive_sentences],
        index=[
            "f10_avg_word_length",
            "f32_double_quotes_count",
            "f54_foreign_letters_count",
            "f76_passive_voice_sentences"
        ]
    )

stylo_cols = [
    "f10_avg_word_length",
    "f32_double_quotes_count",
    "f54_foreign_letters_count",
    "f76_passive_voice_sentences"
]

for split_data in [train_df, val_df, test_df]:
    split_data[stylo_cols] = split_data["raw_text"].apply(extract_assigned_stylometric_features)

# Feature #98: BERT [CLS] Token Probability via AraBERT
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
bert_mlm.eval()
cls_token_id = tokenizer.cls_token_id

@torch.no_grad()
def compute_bert_cls_probability(texts, batch_size=32, max_length=256):
    cls_probs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        outputs = bert_mlm(**encoded)
        cls_logits = outputs.logits[:, 0, :]
        probs = torch.softmax(cls_logits, dim=-1)
        batch_cls_probs = probs[:, cls_token_id].cpu().numpy()
        cls_probs.extend(batch_cls_probs.tolist())
    return np.array(cls_probs, dtype=np.float64)

for name, split_data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"Extracting Feature #98 for {name} set...")
    probs_arr = compute_bert_cls_probability(split_data["raw_text"].tolist())
    split_data["f98_bert_cls_prob"] = probs_arr
    split_data["f98_bert_cls_log10"] = np.log10(np.clip(probs_arr, 1e-30, 1.0))

all_assigned_features = stylo_cols + ["f98_bert_cls_prob"]

# Fit StandardScaler & TF-IDF strictly on Train set
scaler = StandardScaler()
X_train_stylo = scaler.fit_transform(train_df[all_assigned_features])
X_val_stylo = scaler.transform(val_df[all_assigned_features])
X_test_stylo = scaler.transform(test_df[all_assigned_features])

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df["processed_text"])
X_val_tfidf = tfidf.transform(val_df["processed_text"])
X_test_tfidf = tfidf.transform(test_df["processed_text"])

# Save splits, matrices, and transformers
train_df.to_csv("data/processed/train_features.csv", index=False)
val_df.to_csv("data/processed/val_features.csv", index=False)
test_df.to_csv("data/processed/test_features.csv", index=False)

np.save("data/processed/X_train_stylo.npy", X_train_stylo)
np.save("data/processed/X_val_stylo.npy", X_val_stylo)
np.save("data/processed/X_test_stylo.npy", X_test_stylo)

sparse.save_npz("data/processed/X_train_tfidf.npz", X_train_tfidf)
sparse.save_npz("data/processed/X_val_tfidf.npz", X_val_tfidf)
sparse.save_npz("data/processed/X_test_tfidf.npz", X_test_tfidf)

joblib.dump(scaler, "models/stylometric_scaler.joblib")
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

# Clear Visualization of Assigned Stylometric Features (Train Set)
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
colors = {"Human": "#2b5c8f", "AI": "#d95f02"}

sns.kdeplot(data=train_df, x="f10_avg_word_length", hue="label", fill=True, common_norm=False, palette=colors, ax=axes[0], linewidth=2)
axes[0].set_title("Feature #10: Average Word Length", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Avg Word Length (characters)")

sns.pointplot(data=train_df, x="label", y="f32_double_quotes_count", palette=colors, errorbar=("ci", 95), linestyles="none", capsize=0.12, ax=axes[1])
axes[1].set_title("Feature #32: Double Quotes (Mean ± 95% CI)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("Mean Count")

sns.pointplot(data=train_df, x="label", y="f54_foreign_letters_count", palette=colors, errorbar=("ci", 95), linestyles="none", capsize=0.12, ax=axes[2])
axes[2].set_title("Feature #54: Foreign Letters (Mean ± 95% CI)", fontsize=11, fontweight="bold")
axes[2].set_xlabel("")
axes[2].set_ylabel("Mean Count")

sns.pointplot(data=train_df, x="label", y="f76_passive_voice_sentences", palette=colors, errorbar=("ci", 95), linestyles="none", capsize=0.12, ax=axes[3])
axes[3].set_title("Feature #76: Passive Voice Sentences (Mean ± 95% CI)", fontsize=11, fontweight="bold")
axes[3].set_xlabel("")
axes[3].set_ylabel("Mean Count per Abstract")

sns.kdeplot(data=train_df, x="f98_bert_cls_log10", hue="label", fill=True, common_norm=False, palette=colors, ax=axes[4], linewidth=2)
axes[4].set_title("Feature #98: BERT [CLS] Prob (log10)", fontsize=11, fontweight="bold")
axes[4].set_xlabel("log10(CLS Probability)")

axes[5].axis("off")
plt.suptitle("Phase 3: Clear Comparison of Assigned Stylometric Features (Train Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/phase3_assigned_features.png", dpi=300)
plt.close()
print("Phase 3 completed successfully.")