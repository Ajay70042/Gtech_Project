# Intelligent Netflix Content Analytics Pipeline

## Overview

This project combines **Machine Learning**, **Data Analytics**, and **Generative AI (LLM)** to analyze the Netflix Catalog Dataset.

Instead of only performing statistical analysis, the system:

- Cleans and prepares Netflix metadata.
- Detects unusual (anomalous) titles using Machine Learning.
- Uses a Local Large Language Model (LLM) to generate strategic business insights.
- Allows users to ask natural language questions about the dataset using LangChain.

---

# Project Architecture

The project contains three major components:

1. Data Engineering
2. Machine Learning (Isolation Forest)
3. Generative AI + LangChain Agent

---

# Libraries Used

## 1. NumPy

```python
import numpy as np
```

### Purpose

NumPy provides fast numerical operations.

Although it is not heavily used in this project, it is commonly imported because many data science libraries depend on it internally.

---

## 2. Pandas

```python
import pandas as pd
```

### Purpose

Pandas is the main library used for data manipulation.

Used for:

- Loading CSV files
- Cleaning data
- Creating new columns
- Filtering rows
- Handling missing values

Example:

```python
df = pd.read_csv(...)
```

---

## 3. LangChain Core

```python
from langchain_core.prompts import PromptTemplate
```

### What is LangChain?

LangChain is a framework that connects Large Language Models (LLMs) with external tools such as:

- DataFrames
- Databases
- PDFs
- APIs
- Search engines

It allows the LLM to reason over real-world data instead of relying only on its training.

---

### PromptTemplate

PromptTemplate creates structured prompts dynamically.

Instead of writing a fixed prompt, placeholders are used.

Example:

```python
PromptTemplate.from_template(
"Title: {title}
Genre: {genre}"
)
```

Later:

```python
prompt.invoke({
"title":"Titanic",
"genre":"Drama"
})
```

Result:

```
Title: Titanic
Genre: Drama
```

This makes prompts reusable and organized.

---

## 4. LangChain Experimental

```python
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
```

### Purpose

This creates an AI agent capable of interacting directly with a Pandas DataFrame.

Instead of manually writing pandas code, users can ask questions in plain English.

Example:

> Which country has the most TV Shows?

The agent automatically:

1. Understands the question.
2. Writes pandas operations internally.
3. Executes them.
4. Returns the answer.

This enables natural language interaction with structured data.

---

## 5. LangChain Ollama

```python
from langchain_ollama import ChatOllama
```

### Purpose

This connects LangChain with Ollama to run local Large Language Models.

In this project:

```python
llm = ChatOllama(model="gemma2:2b")
```

The Gemma 2B model runs locally, meaning:

- No internet required
- Data remains private
- No API costs
- Faster inference on local hardware

---

## 6. Isolation Forest

```python
from sklearn.ensemble import IsolationForest
```

### Purpose

Isolation Forest is an unsupervised anomaly detection algorithm.

Unlike classification models, it does not require labeled data.

It identifies observations that differ significantly from the rest of the dataset.

Examples of anomalies include:

- Extremely long movies
- Very old content added recently
- Rare metadata combinations

---

# Step-by-Step Workflow

## Step 1 – Load Dataset

```python
df = pd.read_csv(...)
```

The Netflix dataset is loaded into a Pandas DataFrame.

---

## Step 2 – Feature Engineering

New numerical features are created.

### duration_num

Extracts numerical runtime.

Example:

```
"90 min" → 90
```

---

### added_year

Extracts the year from the `date_added` column.

Example:

```
September 9, 2019
↓

2019
```

---

### platform_delay

Calculates how many years passed before Netflix added the content.

Formula:

```
Platform Delay

= Added Year − Release Year
```

Example:

```
Released: 1998

Added: 2020

Delay = 22 years
```

---

## Step 3 – Machine Learning

Features used:

```
duration_num

release_year

platform_delay
```

Isolation Forest analyzes these numerical attributes.

Each title receives:

- Normal
- Anomaly

Only anomalies are selected.

---

## Step 4 – LLM Strategy Generation

The first anomaly is sent to the LLM.

Prompt contains:

- Title
- Runtime
- Genres
- Country
- Release Year
- Description

The LLM acts as a Content Acquisition Strategist.

Instead of predicting numbers, it explains:

- Why the title is unusual
- Whether similar content should be acquired
- Business implications

This transforms numerical results into executive-level insights.

---

## Step 5 – Create DataFrame Agent

```python
agent = create_pandas_dataframe_agent(...)
```

This allows users to ask questions such as:

- Which country produced the most movies?
- Show all anomalies.
- Average movie duration.
- Top genres.
- Which year had the highest number of releases?

The AI automatically converts these questions into Pandas operations.

---

## Step 6 – Interactive Chat Loop

The program continuously accepts user questions.

Example:

```
User:
Which country has the most movies?
```

↓

LangChain Agent

↓

Pandas

↓

Result

↓

Displayed to the user

The loop continues until the user types:

```
exit
```

or

```
quit
```

---

# Overall System Flow

```
Netflix Dataset
        │
        ▼
 Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Isolation Forest
        │
        ▼
Detected Anomalies
        │
        ▼
PromptTemplate
        │
        ▼
Gemma 2B (Ollama)
        │
        ▼
Business Strategy Brief
        │
        ▼
LangChain DataFrame Agent
        │
        ▼
Natural Language Questions
        │
        ▼
Automatic Pandas Analysis
```

---

# Key Technologies

| Technology | Purpose |
|------------|---------|
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical computing support |
| Scikit-learn | Isolation Forest anomaly detection |
| LangChain Core | Prompt engineering and LLM orchestration |
| PromptTemplate | Dynamic prompt creation |
| LangChain Experimental | AI-powered Pandas DataFrame agent |
| Ollama | Runs local LLMs |
| Gemma 2B | Generates strategic business insights |
| Isolation Forest | Detects unusual catalog entries |

---

# Final Outcome

This project demonstrates an end-to-end AI analytics pipeline that integrates:

- Data Engineering for preprocessing and feature extraction.
- Machine Learning for anomaly detection using Isolation Forest.
- Generative AI to transform detected anomalies into strategic business insights.
- LangChain-powered conversational analytics, enabling users to query the dataset in natural language.

The result is an intelligent decision-support system that combines statistical analysis with LLM-based reasoning to produce actionable content strategy recommendations.
