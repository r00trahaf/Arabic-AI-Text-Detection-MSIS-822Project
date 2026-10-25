
"""
Phase 2: Arabic Text Preprocessing and Exploratory Data Analysis
Project: Detection of AI-Generated Arabic Text
Dataset split: from_title_and_content

The original text is preserved. Preprocessing is performed on a separate
copy because punctuation, diacritics, and Latin characters may be useful
for later stylometric feature engineering.
"""

import re
import warnings
from collections import Counter
from itertools import islice

import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.util import ngrams
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

BASE_DIR = Path("/content")
DATA_PATH = (
    BASE_DIR / "data/processed/"
    "from_title_and_content_classification.csv"
)

OUTPUT_DIR = BASE_DIR / "data/processed"
FIGURE_DIR = BASE_DIR / "reports/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TOP_N = 20

# ---------------------------------------------------------
# 2. Arabic text preprocessing
# ---------------------------------------------------------

nltk.download("stopwords", quiet=True)

stemmer = ISRIStemmer()
arabic_stopwords = set(stopwords.words("arabic"))


def normalize_arabic(text):
    """Normalize common Arabic letter variants and remove diacritics."""
    text = str(text)
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ـ", "")
    return text


normalized_stopwords = {
    normalize_arabic(word) for word in arabic_stopwords
}


def tokenize_arabic(text):
    """Extract Arabic word tokens from normalized text."""
    return re.findall(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]+", text)


def preprocess_arabic(text):
    """
    Normalize Arabic, tokenize, remove standard Arabic stopwords,
    and apply ISRI stemming.
    """
    normalized_text = normalize_arabic(text)
    tokens = tokenize_arabic(normalized_text)
    tokens = [
        token for token in tokens
        if token not in normalized_stopwords
    ]
    stems = [stemmer.stem(token) for token in tokens]
    return " ".join(stems)


# ---------------------------------------------------------
# 3. Descriptive statistics from ORIGINAL text
# ---------------------------------------------------------

def calculate_text_statistics(text):
    """
    Calculate descriptive and stylometric statistics.
    The input is the original, unmodified text.
    """
    text = str(text)
    words = text.split()
    word_lengths = [len(word) for word in words]

    # Approximate sentence segmentation based on punctuation.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?؟؛]+", text)
        if sentence.strip()
    ]

    sentence_word_counts = [
        len(sentence.split()) for sentence in sentences
    ]

    unique_words = set(words)

    return {
        "character_count": len(text),
        "whitespace_word_count": len(words),
        "average_word_length": (
            sum(word_lengths) / len(word_lengths)
            if word_lengths else 0
        ),
        "sentence_count_estimate": len(sentences),
        "average_sentence_length_estimate": (
            sum(sentence_word_counts) / len(sentence_word_counts)
            if sentence_word_counts else 0
        ),
        "type_token_ratio": (
            len(unique_words) / len(words) if words else 0
        ),
        "latin_letter_count": len(re.findall(r"[A-Za-z]", text)),
        "double_quote_count": text.count('"'),
        "single_quote_count": text.count("'"),
        "comma_count": text.count(",") + text.count("،"),
        "period_count": text.count("."),
        "question_mark_count": text.count("?") + text.count("؟"),
        "exclamation_mark_count": text.count("!"),
        "semicolon_count": text.count(";") + text.count("؛"),
        "tatweel_count": text.count("ـ"),
        "diacritic_count": len(
            re.findall(r"[\u064B-\u065F\u0670]", text)
        ),
        "line_count": len(text.splitlines()),
        "paragraph_count": len(
            [part for part in re.split(r"\n\s*\n", text.strip()) if part]
        ),
    }


# ---------------------------------------------------------
# 4. Frequency analysis helpers
# ---------------------------------------------------------

def get_tokens(text):
    """Get normalized Arabic tokens for lexical analysis."""
    return tokenize_arabic(normalize_arabic(text))


def frequency_table(texts, n=1):
    """Return the most frequent word or n-gram sequences."""
    counter = Counter()

    for text in texts:
        tokens = get_tokens(text)
        counter.update(ngrams(tokens, n))

    return pd.DataFrame(
        [
            {"ngram": " ".join(gram), "frequency": count}
            for gram, count in counter.most_common(TOP_N)
        ],
        columns=["ngram", "frequency"],
    )


def plot_comparison(table, title, filename):
    """Plot frequency tables by class."""
    if table.empty:
        return

    pivot = table.pivot(
        index="ngram", columns="label", values="frequency"
    ).fillna(0)

    pivot = pivot.sort_values(
        by=list(pivot.columns), ascending=False
    ).head(TOP_N)

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Word or n-gram")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close()


# ---------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------

def save_distribution_plot(df, column, title, filename):
    """Compare a numerical feature across Human and AI classes."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, group in df.groupby("label"):
        ax.hist(
            group[column].dropna(),
            bins=30,
            alpha=0.5,
            label=str(label),
        )

    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Number of abstracts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close(fig)


def save_wordcloud(texts, title, filename):
    """Save a word cloud when the installed font supports Arabic."""
    tokens = [
        token
        for text in texts
        for token in get_tokens(text)
    ]
    if not tokens:
        return

    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        cloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            font_path=font_path,
            collocations=False,
            random_state=RANDOM_SEED,
        ).generate(" ".join(tokens))

        plt.figure(figsize=(12, 6))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / filename, dpi=200)
        plt.close()
    except (ValueError, OSError) as error:
        print(f"Word cloud skipped ({title}): {error}")


# ---------------------------------------------------------
# 6. Main workflow
# ---------------------------------------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(
            "Missing text or label values must be investigated first."
        )

    df["text"] = df["text"].astype(str)
    original_text = df["text"].copy()

    print("=" * 60)
    print("PHASE 2: PREPROCESSING AND EDA")
    print("=" * 60)
    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nMissing values:")
    print(df[["text", "label"]].isna().sum())
    print("\nExact duplicate text rows:", df["text"].duplicated().sum())
    print(
        "Duplicate text-label pairs:",
        df.duplicated(subset=["text", "label"]).sum(),
    )

    # Calculate statistics before preprocessing.
    stats = pd.DataFrame(
        df["text"].apply(calculate_text_statistics).tolist()
    )
    stats.insert(0, "label", df["label"].values)

    stats_path = OUTPUT_DIR / "phase2_text_statistics.csv"
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    summary = stats.groupby("label").agg(["mean", "median", "std"])
    summary_path = OUTPUT_DIR / "phase2_statistics_by_class.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    print("\nDescriptive statistics by class:")
    print(summary.round(3))

    # Keep the original text and store preprocessing separately.
    df["text_processed"] = df["text"].apply(preprocess_arabic)

    assert df["text"].equals(original_text), (
        "The original text column was unexpectedly modified."
    )

    processed_path = (
        OUTPUT_DIR / "from_title_and_content_preprocessed.csv"
    )
    df[["text", "text_processed", "label"]].to_csv(
        processed_path, index=False, encoding="utf-8-sig"
    )

    print("\nPreprocessed dataset saved:", processed_path)
    print("Original text preserved:", df["text"].equals(original_text))

    # Word, bigram, and trigram frequencies by class.
    for n, name in [(1, "unigram"), (2, "bigram"), (3, "trigram")]:
        class_tables = []

        for label, group in df.groupby("label"):
            table = frequency_table(group["text"], n=n)
            table["label"] = label
            class_tables.append(table)

        combined = pd.concat(class_tables, ignore_index=True)
        combined_path = (
            OUTPUT_DIR / f"phase2_{name}_frequencies.csv"
        )
        combined.to_csv(
            combined_path, index=False, encoding="utf-8-sig"
        )

        plot_comparison(
            combined,
            f"Most Frequent {name.title()}s by Class",
            f"phase2_{name}_comparison.png",
        )

    # Function-word analysis on normalized tokens.
    function_word_rows = []

    for label, group in df.groupby("label"):
        counts = Counter()

        for text in group["text"]:
            counts.update(
                token for token in get_tokens(text)
                if token in normalized_stopwords
            )

        for word, count in counts.most_common(TOP_N):
            function_word_rows.append({
                "label": label,
                "word": word,
                "frequency": count,
            })

    function_words = pd.DataFrame(function_word_rows)
    function_words.to_csv(
        OUTPUT_DIR / "phase2_function_word_frequencies.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not function_words.empty:
        plot_comparison(
            function_words.rename(columns={"word": "ngram"}),
            "Common Arabic Function Words by Class",
            "phase2_function_words_comparison.png",
        )

    # Punctuation and text-length comparisons.
    plot_columns = [
        ("character_count", "Text Character Count"),
        ("whitespace_word_count", "Word Count"),
        ("average_word_length", "Average Word Length"),
        ("sentence_count_estimate", "Estimated Sentence Count"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("diacritic_count", "Arabic Diacritic Count"),
        ("double_quote_count", "Double Quotation Marks"),
    ]

    for column, title in plot_columns:
        save_distribution_plot(
            stats, column, f"{title} by Class",
            f"phase2_{column}_distribution.png",
        )

    # Word clouds for each class.
    for label, group in df.groupby("label"):
        safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", str(label))
        save_wordcloud(
            group["text"],
            f"Word Cloud: {label}",
            f"phase2_wordcloud_{safe_label}.png",
        )

    # Save a compact class-level summary of key descriptive features.
    key_features = [
        "character_count",
        "whitespace_word_count",
        "average_word_length",
        "sentence_count_estimate",
        "average_sentence_length_estimate",
        "type_token_ratio",
    ]
    stats.groupby("label")[key_features].mean().round(3).to_csv(
        OUTPUT_DIR / "phase2_key_eda_summary.csv",
        encoding="utf-8-sig",
    )

    print("\nPhase 2 completed.")
    print("Statistics and processed data:", OUTPUT_DIR)
    print("Figures:", FIGURE_DIR)
    print(
        "\nNote: Sentence counts are estimates based on punctuation. "
        "Word clouds may require an Arabic-compatible font or shaping."
    )


if __name__ == "__main__":
    main()
