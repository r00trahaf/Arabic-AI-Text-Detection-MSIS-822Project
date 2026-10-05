# ============================================================
# MSIS-822 Final Project
# Detection of AI-Generated Arabic Text
#
# Phase 1: Project Setup & Data Acquisition
#
# Dataset:
# KFUPM-JRCAI/arabic-generated-abstracts
#
# Selected subset:
# from_title_and_content
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

from pathlib import Path

import pandas as pd
from datasets import load_dataset


# ============================================================
# 2. Configuration
# ============================================================

DATASET_NAME = "KFUPM-JRCAI/arabic-generated-abstracts"
SELECTED_SPLIT = "from_title_and_content"

BASE_PATH = Path("/content")

RAW_DATA_PATH = BASE_PATH / "data" / "raw"
PROCESSED_DATA_PATH = BASE_PATH / "data" / "processed"

RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Load Dataset from Hugging Face
# ============================================================

print("=" * 70)
print("Loading dataset from Hugging Face")
print("=" * 70)

dataset = load_dataset(DATASET_NAME)

print(dataset)


# ============================================================
# 4. Inspect Dataset Structure
# ============================================================

print("\n" + "=" * 70)
print("Dataset Structure")
print("=" * 70)

print(f"Available subsets: {list(dataset.keys())}")

for split_name in dataset.keys():
    print(f"\nSubset: {split_name}")
    print(f"Number of rows: {len(dataset[split_name])}")
    print(f"Features: {dataset[split_name].features}")


# ============================================================
# 5. Select the Required Subset
# ============================================================

print("\n" + "=" * 70)
print(f"Selected subset: {SELECTED_SPLIT}")
print("=" * 70)

selected_dataset = dataset[SELECTED_SPLIT]

print(f"Number of raw samples: {len(selected_dataset)}")
print(f"Features: {selected_dataset.features}")


# ============================================================
# 6. Convert Selected Subset to Pandas DataFrame
# ============================================================

df_raw = selected_dataset.to_pandas()

print("\nRaw DataFrame shape:")
print(df_raw.shape)

print("\nColumns:")
print(df_raw.columns.tolist())

print("\nData types:")
print(df_raw.dtypes)


# ============================================================
# 7. Save Original Raw Data
# ============================================================

raw_file = RAW_DATA_PATH / f"{SELECTED_SPLIT}.csv"

df_raw.to_csv(
    raw_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nRaw data saved to:")
print(raw_file)


# ============================================================
# 8. Data Quality Checks BEFORE Reshaping
# ============================================================

print("\n" + "=" * 70)
print("RAW DATA QUALITY CHECKS")
print("All checks are performed before reshaping.")
print("=" * 70)


# ------------------------------------------------------------
# 8.1 Missing Values
# ------------------------------------------------------------

print("\n--- Missing Values ---")

missing_values = df_raw.isnull().sum()

print(missing_values)

print(
    f"\nTotal missing values: "
    f"{missing_values.sum()}"
)


# ------------------------------------------------------------
# 8.2 Duplicate Rows
# ------------------------------------------------------------

print("\n--- Duplicate Rows ---")

duplicate_rows = df_raw.duplicated().sum()

print(f"Duplicate rows: {duplicate_rows}")


# ------------------------------------------------------------
# 8.3 Empty Strings
# ------------------------------------------------------------

print("\n--- Empty Text Values ---")

text_columns = [
    "original_abstract",
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract"
]

empty_values = {}

for column in text_columns:
    empty_count = (
        df_raw[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_values[column] = empty_count

    print(f"{column}: {empty_count}")


# ------------------------------------------------------------
# 8.4 Data Types
# ------------------------------------------------------------

print("\n--- Data Types ---")

print(df_raw.dtypes)

non_string_values = {}

for column in text_columns:
    count = (~df_raw[column].map(lambda x: isinstance(x, str))).sum()
    non_string_values[column] = count

print("\nNon-string values:")
for column, count in non_string_values.items():
    print(f"{column}: {count}")


# ------------------------------------------------------------
# 8.5 Basic Text Length Check
# ------------------------------------------------------------

print("\n--- Text Length Check ---")

for column in text_columns:

    lengths = df_raw[column].fillna("").astype(str).str.len()

    print(f"\n{column}")
    print(f"Minimum length: {lengths.min()}")
    print(f"Maximum length: {lengths.max()}")
    print(f"Empty texts: {(lengths == 0).sum()}")


# ============================================================
# 9. Data Quality Summary
# ============================================================

print("\n" + "=" * 70)
print("RAW DATA QUALITY SUMMARY")
print("=" * 70)

print(f"Raw rows: {len(df_raw)}")
print(f"Raw columns: {len(df_raw.columns)}")
print(f"Missing values: {missing_values.sum()}")
print(f"Duplicate rows: {duplicate_rows}")

total_empty_values = sum(empty_values.values())

print(f"Empty text values: {total_empty_values}")


# ============================================================
# 10. Reshape the Dataset
# ============================================================

print("\n" + "=" * 70)
print("RESHAPING DATASET")
print("=" * 70)


# Human-written abstracts
human_df = pd.DataFrame({
    "text": df_raw["original_abstract"],
    "label": "Human"
})


# AI-generated abstracts
ai_columns = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract"
]

ai_parts = []

for column in ai_columns:

    temp = pd.DataFrame({
        "text": df_raw[column],
        "label": "AI"
    })

    ai_parts.append(temp)


ai_df = pd.concat(
    ai_parts,
    ignore_index=True
)


# Combine Human and AI
df_final = pd.concat(
    [human_df, ai_df],
    ignore_index=True
)


# ============================================================
# 11. Inspect Final Classification Dataset
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLASSIFICATION DATASET")
print("=" * 70)

print(f"Final shape: {df_final.shape}")

print("\nColumns:")
print(df_final.columns.tolist())

print("\nFirst five rows:")
print(df_final.head())


# ============================================================
# 12. Target Label Distribution
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

label_counts = df_final["label"].value_counts()

label_percentages = (
    df_final["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCounts:")
print(label_counts)

print("\nPercentages:")
print(label_percentages)


# ============================================================
# 13. Final Dataset Checks
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET CHECKS")
print("=" * 70)

print(
    f"Missing text values: "
    f"{df_final['text'].isnull().sum()}"
)

print(
    f"Missing labels: "
    f"{df_final['label'].isnull().sum()}"
)

print(
    f"Non-string text values: "
    f"{(~df_final['text'].map(lambda x: isinstance(x, str))).sum()}"
)

print(
    f"Texts shorter than 10 characters: "
    f"{(df_final['text'].str.len() < 10).sum()}"
)


# ============================================================
# 14. Text Length Statistics
# ============================================================

print("\n" + "=" * 70)
print("TEXT LENGTH STATISTICS")
print("=" * 70)

text_lengths = df_final["text"].str.len()

print(text_lengths.describe())


# ============================================================
# 15. Save Final Classification Dataset
# ============================================================

processed_file = (
    PROCESSED_DATA_PATH /
    "from_title_and_content_classification.csv"
)

df_final.to_csv(
    processed_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED")
print("=" * 70)

print(f"Raw dataset:")
print(raw_file)

print(f"\nFinal classification dataset:")
print(processed_file)

print(f"\nFinal number of rows: {len(df_final)}")
print(f"Final number of columns: {len(df_final.columns)}")
