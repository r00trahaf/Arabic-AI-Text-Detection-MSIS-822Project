```python
"""
MSIS-822 Advanced Data Analytic Techniques

Project:
Detection of AI-Generated Arabic Text: A Data Mining Approach

Phase 1:
Project Setup & Data Acquisition
Initial Data Exploration

This script:
1. Downloads the dataset from Hugging Face.
2. Saves the original data as raw CSV files.
3. Loads the raw CSV files.
4. Reshapes the data into Human vs AI classification format.
5. Combines all dataset subsets.
6. Performs initial data-quality checks:
   - Dataset structure
   - Label distribution
   - Missing values
   - Duplicate texts
   - Label inconsistencies
   - Data types
   - Non-string values
   - Text length
   - Very short texts
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_NAME = "KFUPM-JRCAI/arabic-generated-abstracts"

RAW_DATA_PATH = Path("/content/data/raw")


# ============================================================
# 3. DOWNLOAD DATASET FROM HUGGING FACE
# ============================================================

dataset = load_dataset(DATASET_NAME)

print("Dataset loaded successfully:")
print(dataset)


# ============================================================
# 4. INITIAL DATASET STRUCTURE
# ============================================================

print("\nDataset splits:")
for split_name in dataset.keys():
    print(
        f"{split_name}: "
        f"{len(dataset[split_name])} rows"
    )

print("\nDataset features:")
for split_name in dataset.keys():
    print(f"\n{split_name}:")
    print(dataset[split_name].features)


# ============================================================
# 5. SAVE RAW DATA
# ============================================================

RAW_DATA_PATH.mkdir(
    parents=True,
    exist_ok=True
)

dataset["by_polishing"].to_pandas().to_csv(
    RAW_DATA_PATH / "by_polishing.csv",
    index=False
)

dataset["from_title"].to_pandas().to_csv(
    RAW_DATA_PATH / "from_title.csv",
    index=False
)

dataset["from_title_and_content"].to_pandas().to_csv(
    RAW_DATA_PATH / "from_title_and_content.csv",
    index=False
)

print("\nRaw data saved successfully.")

print("\nRaw files:")
print([file.name for file in RAW_DATA_PATH.iterdir()])


# ============================================================
# 6. LOAD RAW CSV FILES
# ============================================================

df_by_polishing = pd.read_csv(
    RAW_DATA_PATH / "by_polishing.csv"
)

df_from_title = pd.read_csv(
    RAW_DATA_PATH / "from_title.csv"
)

df_from_title_and_content = pd.read_csv(
    RAW_DATA_PATH / "from_title_and_content.csv"
)


# ============================================================
# 7. CHECK SHAPES OF RAW DATA
# ============================================================

print("\nRaw dataset shapes:")

print(
    "by_polishing:",
    df_by_polishing.shape
)

print(
    "from_title:",
    df_from_title.shape
)

print(
    "from_title_and_content:",
    df_from_title_and_content.shape
)


# ============================================================
# 8. INITIAL MISSING VALUES AND DUPLICATES CHECK
# ============================================================

print("\nMissing values in raw datasets:")

print("\nby_polishing:")
print(df_by_polishing.isna().sum())

print("\nfrom_title:")
print(df_from_title.isna().sum())

print("\nfrom_title_and_content:")
print(df_from_title_and_content.isna().sum())


print("\nDuplicate rows in raw datasets:")

print(
    "by_polishing:",
    df_by_polishing.duplicated().sum()
)

print(
    "from_title:",
    df_from_title.duplicated().sum()
)

print(
    "from_title_and_content:",
    df_from_title_and_content.duplicated().sum()
)


# ============================================================
# 9. PREPARE HUMAN AND AI DATA
# ============================================================

def prepare_split(df, method_name):
    """
    Convert one dataset split into Human vs AI format.

    The original abstract is labeled Human.
    The four generated abstracts are labeled AI.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset split.
    method_name : str
        Generation method name.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        text, label, generation_method
    """

    # Human-written abstracts
    human = pd.DataFrame({
        "text": df["original_abstract"],
        "label": "Human",
        "generation_method": method_name
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
            "text": df[column],
            "label": "AI",
            "generation_method": method_name
        })

        ai_parts.append(temp)

    ai = pd.concat(
        ai_parts,
        ignore_index=True
    )

    return pd.concat(
        [human, ai],
        ignore_index=True
    )


# ============================================================
# 10. PREPARE EACH DATASET SPLIT
# ============================================================

df1 = prepare_split(
    df_by_polishing,
    "by_polishing"
)

df2 = prepare_split(
    df_from_title,
    "from_title"
)

df3 = prepare_split(
    df_from_title_and_content,
    "from_title_and_content"
)


print("\nPrepared split shapes:")

print(
    "by_polishing:",
    df1.shape
)

print(
    "from_title:",
    df2.shape
)

print(
    "from_title_and_content:",
    df3.shape
)


# ============================================================
# 11. COMBINE ALL DATASET SPLITS
# ============================================================

df = pd.concat(
    [df1, df2, df3],
    ignore_index=True
)

print("\nCombined dataset shape:")
print(df.shape)


# ============================================================
# 12. LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel distribution (%):")
print(
    df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 13. GENERATION METHOD DISTRIBUTION
# ============================================================

print("\nGeneration method distribution:")
print(
    df["generation_method"].value_counts()
)


# ============================================================
# 14. MISSING AND EMPTY TEXT CHECK
# ============================================================

empty_texts = (
    df["text"].isna()
    | (df["text"].str.strip() == "")
)

print(
    "\nEmpty or whitespace-only texts:",
    empty_texts.sum()
)


print(
    "\nMissing labels:",
    df["label"].isna().sum()
)


# ============================================================
# 15. DATA TYPES
# ============================================================

print("\nData types:")
print(df.dtypes)


# ============================================================
# 16. NON-STRING TEXT VALUES
# ============================================================

non_string_texts = df["text"].apply(
    lambda x: not isinstance(x, str)
)

print(
    "\nNon-string text values:",
    non_string_texts.sum()
)


# ============================================================
# 17. DUPLICATE TEXT ANALYSIS
# ============================================================

duplicate_texts = df[
    df["text"].duplicated(keep=False)
].copy()

print(
    "\nRows involved in duplicated texts:",
    len(duplicate_texts)
)

print(
    "Unique duplicated texts:",
    duplicate_texts["text"].nunique()
)

print(
    "\nDuplicated rows by label:"
)

print(
    duplicate_texts["label"].value_counts()
)

print(
    "\nDuplicated rows by generation method:"
)

print(
    duplicate_texts["generation_method"].value_counts()
)


# ============================================================
# 18. TEXT DUPLICATION COUNTS
# ============================================================

print(
    "\nTotal repeated text occurrences:",
    df["text"].duplicated().sum()
)

print(
    "Duplicate rows based on text + label:",
    df.duplicated(
        subset=["text", "label"]
    ).sum()
)


# ============================================================
# 19. IDENTIFY TEXTS WITH BOTH HUMAN AND AI LABELS
# ============================================================

conflicting_texts = (
    df.groupby("text")
    .filter(
        lambda x: x["label"].nunique() > 1
    )
)

conflict_summary = (
    conflicting_texts
    .groupby(["text", "label"])
    .size()
    .unstack(fill_value=0)
)

print(
    "\nUnique texts with both Human and AI labels:",
    len(conflict_summary)
)

print("\nConflicting label summary:")
print(conflict_summary)


# ============================================================
# 20. DETAILS OF CONFLICTING TEXTS
# ============================================================

conflict_details = (
    df[
        df["text"].isin(conflict_summary.index)
    ]
    .sort_values(
        ["text", "label", "generation_method"]
    )
)

print("\nConflicting text details:")
print(
    conflict_details[
        ["text", "label", "generation_method"]
    ]
)


# ============================================================
# 21. TEXT LENGTH ANALYSIS
# ============================================================

df["text_length"] = df["text"].str.len()

print("\nText length statistics:")
print(
    df["text_length"].describe()
)


# ============================================================
# 22. VERY SHORT TEXT CHECK
# ============================================================

short_texts = df[
    df["text_length"] < 10
]

print(
    "\nTexts shorter than 10 characters:",
    len(short_texts)
)


# ============================================================
# 23. UNIQUE LABEL VALUES
# ============================================================

print("\nUnique labels:")
print(
    df["label"].unique()
)


print(
    "\nMissing labels:"
)

print(
    df["label"].isna().sum()
)


# ============================================================
# 24. FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL PHASE 1 DATASET SUMMARY")
print("=" * 60)

print(
    f"Total records: {len(df):,}"
)

print(
    f"Human records: {(df['label'] == 'Human').sum():,}"
)

print(
    f"AI records: {(df['label'] == 'AI').sum():,}"
)

print(
    f"Unique texts: {df['text'].nunique():,}"
)

print(
    f"Missing values: {df.isna().sum().sum():,}"
)

print(
    f"Empty/whitespace texts: {empty_texts.sum():,}"
)

print(
    f"Non-string texts: {non_string_texts.sum():,}"
)

print(
    f"Texts shorter than 10 characters: {len(short_texts):,}"
)

print(
    f"Unique texts with conflicting labels: "
    f"{len(conflict_summary):,}"
)

print("=" * 60)
```
