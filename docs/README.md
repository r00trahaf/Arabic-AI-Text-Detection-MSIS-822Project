# Arabic AI Text Detection (MSIS-822 Project)

A machine learning project focused on detecting and classifying AI-generated Arabic text across multiple Large Language Models (such as ALLAM, LLaMA, and OpenAI) using the `KFUPM-JRCAI/arabic-generated-abstracts` dataset.

## Project Overview
With the rapid advancements in generative AI, distinguishing between human-written and machine-generated Arabic text has become critical. This project implements an end-to-end data science pipeline—from raw data acquisition and exploratory data analysis (EDA) to feature engineering, model training, and evaluation.

## Dataset Information
- **Source**: [KFUPM-JRCAI/arabic-generated-abstracts](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts)
- **Total Samples**: 8,388 samples across three distinct subsets:
  - `by_polishing`
  - `from_title`
  - `from_title_and_content`
- **Features**: Contains original human-written abstracts (`original_abstract`) alongside respective AI-generated outputs.
