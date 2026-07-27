# Intelligent Netflix Content Analytics Pipeline

**Prepared & Submitted By:** AJAY BASKEY & PRAKHAR JAIN


This project presents an **intelligent content analytics pipeline** for streaming platforms using the **Netflix Catalog Dataset (`netflix_titles.csv`)**. By combining **data engineering**, **unsupervised machine learning**, and **local Generative AI**, the system transforms raw entertainment metadata into actionable strategic business insights. The pipeline not only detects unusual content within the catalog but also provides AI-generated business recommendations and enables natural language interaction with the dataset.

---

# 📊 Part 1: Data Analytics & Engineering

## Data Cleaning & Optimization
- Loads the raw Netflix metadata into a Pandas DataFrame.
- Cleans missing or inconsistent values to improve data quality.
- Optimizes memory usage by converting repetitive text-based columns (such as `type`, `rating`, and `country`) into **categorical** data types, reducing memory consumption and improving processing efficiency.

## Feature Engineering
- Extracts numerical runtime values by converting duration strings (e.g., `"90 min"`) into numeric values (`90.0`).
- Extracts the year from the `date_added` column to create `added_year`.
- Computes **platform_delay**, representing the number of years between a title's original release and its addition to Netflix.

**Formula:**

```text
platform_delay = added_year - release_year
```

These engineered numerical features enable advanced analytics and machine learning models to process the dataset effectively.

## Dataset Export
- Generates a cleaned and enriched dataset named **`netflix_titles_cleaned.csv`**.
- Preserves both the original metadata and newly engineered features for future analysis, visualization, and machine learning tasks.

---

# 🤖 Part 2: Machine Learning & Generative AI

## ML Outlier Detection
- Trains an **Isolation Forest** model on numerical features including:
  - `duration_num`
  - `release_year`
  - `platform_delay`
- Automatically identifies approximately the **top 1% most unusual titles** in the Netflix catalog based on runtime, content age, and acquisition delay.
- Labels detected outliers as anomalies for further business analysis.

## AI Content Strategy Briefs
- Integrates a locally hosted **Large Language Model (LLM)** using **Ollama** with the lightweight **Gemma2:2B** model.
- Passes metadata from detected anomalies into a structured prompt.
- Generates a concise **three-bullet-point executive strategy brief** explaining:
  - Why the title is considered anomalous.
  - Which characteristics make it unique within the catalog.
  - Whether acquiring similar content could provide strategic market value.

This combines machine learning with generative AI to convert statistical anomalies into meaningful business insights.

## Autonomous Conversational Agent
- Implements a **LangChain Text-to-Pandas Agent** for conversational data analysis.
- Translates plain English questions into executable Pandas code automatically.
- Executes the generated code on the dataset and returns accurate results in real time.

**Example Queries**
- *How many titles were flagged as anomalies?*
- *Show the oldest movies in the catalog.*
- *Which country has produced the most TV Shows?*
- *List movies longer than 180 minutes.*
- *What is the average platform delay?*

This allows users to explore the dataset interactively without writing any Python or SQL code.

---

# 🚀 Project Highlights

- Cleans and optimizes raw Netflix metadata for efficient analysis.
- Engineers meaningful numerical features from unstructured data.
- Detects anomalous content using the **Isolation Forest** algorithm.
- Generates AI-powered strategic business insights with a local **Gemma2:2B** model.
- Integrates **LangChain** to enable conversational querying of structured data.
- Supports interactive, real-time exploration of the Netflix catalog through natural language.
- Demonstrates an end-to-end workflow that combines **Data Engineering**, **Machine Learning**, and **Generative AI** into a unified analytics pipeline.