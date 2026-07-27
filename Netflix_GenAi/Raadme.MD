# AI & Machine Learning Pipeline for Netflix Content Analytics

This project implements an **end-to-end AI and Machine Learning pipeline** for analyzing the Netflix catalog (`netflix_titles.csv`). It combines **feature engineering**, **unsupervised anomaly detection**, **local Large Language Model (LLM) reasoning**, and a **conversational data analysis agent** running locally with **Ollama (gemma2:2b)**. The system identifies unusual content in the dataset and enables users to explore the data using natural language.

---

# 🛠️ Detailed Component Breakdown

## 1. Data Processing & Feature Engineering

The pipeline begins by loading the raw Netflix metadata into a Pandas DataFrame.

### Data Ingestion
- Loads the `netflix_titles.csv` dataset.
- Performs preprocessing and handles missing values where necessary.

### Feature Engineering

Several numerical features are extracted because machine learning models cannot directly process raw text.

#### Runtime Extraction (`duration_num`)
Converts duration strings such as:

- `"90 min"` → `90`
- `"120 min"` → `120`

into numerical values.

#### Content Addition Year (`added_year`)
Extracts the year from the `date_added` column.

Example:

- `"September 9, 2021"` → `2021`

#### Platform Delay (`platform_delay`)
Calculates how long it took for content to appear on Netflix.

Formula:

```text
platform_delay = added_year - release_year
```

Example:

| Release Year | Added Year | Platform Delay |
|--------------|------------|----------------|
| 1995 | 2020 | 25 years |

This feature helps identify unusually old content that was acquired much later.

---

# 2. Machine Learning Anomaly Detection

The project applies an **Isolation Forest**, an unsupervised anomaly detection algorithm.

### Model

```text
IsolationForest(
    contamination=0.01
)
```

### Input Features

The model trains using:

- `duration_num`
- `release_year`
- `platform_delay`

### Purpose

Rather than predicting categories, the model isolates titles whose numerical characteristics differ significantly from the rest of the Netflix catalog.

Examples of anomalies include:

- Extremely long movies
- Very short content
- Very old titles
- Titles added decades after release

With:

```text
contamination = 0.01
```

the model flags approximately the **top 1% most unusual titles**.

---

# 3. Local LLM & Strategic Diagnostic Briefs

After anomaly detection, the project uses a **local Large Language Model** running through **Ollama**.

### Local Inference

Model used:

```text
gemma2:2b
```

running entirely on the user's machine.

### Prompt Engineering

Metadata from the most anomalous title is inserted into a structured prompt.

Example information supplied:

- Title
- Release Year
- Duration
- Platform Delay
- Content Type

### LLM Role

The model acts as a:

> **Senior Content Acquisition Strategist**

and produces a concise three-point strategic report explaining:

- Why the title appears anomalous
- Which characteristics make it unusual
- Whether acquiring similar content is strategically beneficial

This transforms numerical anomaly detection into business-oriented insights.

---

# 4. Cross-Architecture Integration & Interactive Agent

The machine learning output is merged back into the dataset.

### Data Fusion

A new boolean column is added:

```python
df["is_anomaly"]
```

This indicates whether each title was classified as an anomaly.

Example:

| Title | is_anomaly |
|--------|------------|
| Movie A | False |
| Movie B | True |

---

## Conversational Pandas Agent

The project initializes LangChain's:

```python
create_pandas_dataframe_agent()
```

which allows users to interact with the dataset using natural language.

Example questions include:

- How many titles were flagged as anomalies?
- Show the oldest movies.
- Which country has the most TV Shows?
- List movies longer than 180 minutes.
- What is the average platform delay?

The agent automatically:

1. Understands the user's question.
2. Converts it into Pandas code.
3. Executes the code.
4. Returns the answer.

No manual coding is required from the user.

---

## Interactive CLI Loop

The application continuously accepts user queries using a terminal interface.

Example workflow:

```text
User:
How many anomaly titles exist?

↓

LLM Agent

↓

Generated Pandas Code

↓

Dataset Execution

↓

Answer Returned
```

The session continues until the user types:

```text
exit
```

or

```text
quit
```

---

# ⚙️ Overall Architectural Pipeline

```text
                   netflix_titles.csv
                           │
                           ▼
        ┌────────────────────────────────────┐
        │ Data Cleaning & Feature Engineering │
        │                                    │
        │ • duration_num                     │
        │ • added_year                       │
        │ • platform_delay                   │
        └────────────────────────────────────┘
                           │
                           ▼
        ┌────────────────────────────────────┐
        │      Isolation Forest Model        │
        │                                    │
        │ Detects Top 1% Anomalous Titles    │
        └────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
 ┌──────────────────────┐      ┌────────────────────────┐
 │ Add df['is_anomaly'] │      │  Top Anomaly Metadata  │
 └──────────────────────┘      └────────────────────────┘
            │                             │
            │                             ▼
            │              ┌────────────────────────────┐
            │              │ Local LLM (Gemma2:2B)      │
            │              │ Strategic Diagnostic Brief │
            │              └────────────────────────────┘
            │
            ▼
 ┌────────────────────────────────────────────┐
 │ LangChain Pandas DataFrame Agent           │
 │                                            │
 │ Natural Language → Pandas Code → Answer    │
 └────────────────────────────────────────────┘
            │
            ▼
    Interactive Command-Line Interface (CLI)
```

---

# Key Technologies Used

| Component | Technology |
|----------|------------|
| Data Processing | Pandas |
| Feature Engineering | Python |
| Machine Learning | Scikit-learn (Isolation Forest) |
| Local LLM | Ollama (Gemma2:2B) |
| Prompt Engineering | LangChain PromptTemplate |
| AI Framework | LangChain |
| Conversational Data Analysis | Pandas DataFrame Agent |
| User Interface | Interactive CLI |

---

# Project Highlights

- Performs automated feature engineering on raw Netflix metadata.
- Detects anomalous content using the Isolation Forest algorithm.
- Generates business-oriented strategic insights with a local LLM.
- Combines machine learning predictions with conversational AI.
- Enables natural language querying of structured data through LangChain.
- Runs entirely on a local machine without requiring cloud-based inference.
- Demonstrates an integrated AI pipeline that blends classical machine learning with modern generative AI.
