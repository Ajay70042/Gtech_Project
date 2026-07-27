

# **Part 1️⃣: Text-to-Pandas Agents (10 Questions)**

### **Q1. When querying a text-heavy dataset like Netflix Titles, why is a Pandas Agent often more accurate for aggregations (e.g., "count the titles") than a Vector Database (RAG) approach?**
* **Answer:** Vector Databases rely on semantic similarity search over text embeddings, which retrieve approximate document chunks rather than evaluating every record in a dataset. They struggle with exact mathematical counting because they only return top-$K$ semantic matches. A Pandas Agent generates executable Python code that runs deterministic, exact aggregations (e.g., `.shape[0]` or `.count()`) across the entire DataFrame.

---

### **Q2. Why is setting the LLM `temperature` to 0.0 critical when the agent is generating code to filter by specific directors or genres?**
* **Answer:** A temperature of `0.0` eliminates random sampling and forces the LLM to output deterministic, highest-probability tokens. For code generation (such as Pandas indexing or filtering syntax), variability leads to syntax errors, incorrect column names, or broken logic. Zero temperature ensures consistent, reproducible Python code execution.

---

### **Q3. Describe the "Observation" step in the LangChain agent's execution loop. What happens if the agent's initial code throws a 'KeyError' on the dataframe?**
* **Answer:** In the LangChain ReAct loop, the **Observation** is the direct output or error returned from running the LLM's generated code in the Python environment. If the code throws a `KeyError`, the traceback is captured as the Observation and fed back into the LLM's context window. The agent reads the exception, inspects the schema, corrects the column name, and generates a revised script in the next iteration.

---

### **Q4. How does the Pandas agent handle context window limits when dealing with 8,000+ long synopsis strings? What data does it actually send to the OpenAI API?**
* **Answer:** The Pandas Agent does **not** send the 8,000+ full rows or synopsis strings to the API. Instead, it sends the DataFrame schema, data types, and a small preview of the first few rows generated via `df.head()` in the system prompt. The LLM uses this metadata to write local Python code, while code execution over the full dataset happens entirely within the local environment.

---

### **Q5. A user asks the agent to "Change all TV-MA ratings to R". How do you implement guardrails to ensure the agent operates in a read-only capacity?**
* **Answer:** 
* **Copy Isolation:** Pass a deep copy (`df.copy()`) or read-only view to the execution environment.
* **System Prompt Restrictions:** Explicitly instruct the agent in the system prompt that it is strictly forbidden from mutating data or calling assignment operators.
* **REPL AST Filtering:** Intercept generated Python code using AST analysis or custom tool wrappers to block mutating methods (such as `.drop()`, `.rename()`, or `inplace=True`).

---

### **Q6. How does the `handle_parsing_errors=True` parameter prevent the agent application from crashing during unexpected LLM formatting?**
* **Answer:** When an LLM outputs raw text or formatting that fails to strictly adhere to LangChain's expected ReAct JSON/Action structure, `handle_parsing_errors=True` catches the parsing exception internally. It passes the parsing error back to the model as an Observation prompt, allowing the LLM to self-correct its output syntax without crashing the application process.

---

### **Q7. If the agent repeatedly fails to properly split comma-separated lists in the `cast` column, how can you use the System Prompt to guide its Pandas methodology?**
* **Answer:** You inject explicit instructions into the agent's System Prompt (via the `prefix` or `suffix` arguments). For example:
  > *"When performing operations on comma-separated list columns like 'cast', always use `.str.split(', ')` followed by `.explode()` to unnest individual items before grouping or counting."*

---

### **Q8. Explain "Few-Shot Prompting". How would you use it to teach the agent to distinguish between "Seasons" for TV Shows and "Minutes" for Movies when querying durations?**
* **Answer:** **Few-Shot Prompting** provides explicit input-output demonstration pairs within the prompt to teach the model expected reasoning formats.
* **Example Prompt Demonstrations:**
  * *User Prompt:* "Find movies longer than 90 min."  
    *Generated Code:* `df[(df['type'] == 'Movie') & (df['duration'].str.extract(r'(\d+)').astype(float) > 90)]`
  * *User Prompt:* "Find TV shows with more than 3 seasons."  
    *Generated Code:* `df[(df['type'] == 'TV Show') & (df['duration'].str.extract(r'(\d+)').astype(float) > 3)]`

---

### **Q9. Describe the security risks of deploying this agent in a public web application, specifically regarding prompt injection attacks aiming to execute arbitrary Python.**
* **Answer:** Deploying a Pandas Agent requires a Python REPL tool that executes generated code dynamically. Adversaries can perform **Prompt Injection** (e.g., instructing the agent to execute `import os; os.system('...')`), leading to Arbitrary Code Execution (RCE) on the host server, environment variable exfiltration (including API keys), or unauthorized system file access.

---

### **Q10. How would you implement tracing (e.g., using LangSmith or a CallbackHandler) to monitor token usage and the exact Pandas scripts generated by the agent over time?**
* **Answer:** 
* **LangSmith:** Configure environment variables (`LANGCHAIN_TRACING_V2="true"` and `LANGCHAIN_API_KEY="..."`) to automatically log every ReAct step, prompt/completion token count, and generated script execution history.
* **CallbackHandler:** Pass a custom `StdOutCallbackHandler` or `OpenAICallbackHandler` instance into the agent's `callbacks` list during instantiation to inspect generated code strings and token usage metadata programmatically.

# **Part 2️⃣: Automated Anomaly Reasoning (10 Questions)**

# Q11. Why must we extract numeric features (like `duration_num` and `platform_delay`) before utilizing an Isolation Forest for anomaly detection?

**Answer:**

Isolation Forest is a distance- and split-based decision tree algorithm that partitions numerical feature spaces using random splits (hyperplanes). It cannot directly process raw, unformatted text fields such as `"90 min"`, `"3 Seasons"`, or date strings. By extracting and engineering continuous numerical features, the algorithm can mathematically calculate split depths and path lengths, which are essential for assigning anomaly scores.

---

# Q12. What is LLM "Hallucination" in the context of content strategy? How might an LLM invent a false historical context for an anomaly from 1940?

**Answer:**

LLM hallucination occurs when the model generates plausible-sounding but factually incorrect information that is not supported by the provided data. For example, when analyzing a 1940 movie anomaly, the LLM might fabricate historical events such as:

- Non-existent film studio mandates
- Fake box office statistics
- Fictional government censorship policies
- Imaginary production restrictions

These invented explanations may sound convincing but are not grounded in the dataset.

---

# Q13. In the provided `diagnostic_prompt`, what is the specific advantage of passing the `description` (synopsis) alongside the numerical data?

**Answer:**

The `description` provides qualitative and semantic context that numerical features such as `duration_num` and `release_year` cannot capture.

While Machine Learning identifies **that** a title is structurally anomalous, the synopsis helps the LLM understand **why** it may be unusual by analyzing its content. This enables the model to determine whether the anomaly is due to:

- An experimental short film
- A niche documentary
- A rare archival release
- A localized acquisition
- Another unique content characteristic

---

# Q14. Describe how to enforce strict JSON output from the LLM (e.g., using Pydantic) so the Content Strategy Brief can be automatically inserted into a dashboard.

**Answer:**

Define a structured schema using **Pydantic** by creating a `BaseModel` with explicit field definitions.

```python
from pydantic import BaseModel, Field

class ContentStrategyBrief(BaseModel):
    anomaly_reason: str = Field(
        description="Primary structural reason for anomaly"
    )
    strategic_value: str = Field(
        description="Analysis of market/niche value"
    )
    recommendation: str = Field(
        description="Actionable acquisition advice"
    )
```

Then invoke the LLM using methods such as:

- `with_structured_output(ContentStrategyBrief)`
- `PydanticOutputParser`

This guarantees that the response conforms to a valid JSON schema, making it suitable for automatic ingestion into dashboards or downstream systems.

---

# Q15. What is the functional difference between a `SystemMessage` establishing the "Acquisition Strategist" persona and the `HumanMessage` containing the actual anomaly data?

**Answer:**

### SystemMessage

Defines the model's permanent behavior, expertise, constraints, and response style.

Example:

> "You are a Senior Content Acquisition Strategist responsible for evaluating unusual catalog titles."

It tells the model **how to think and respond**.

### HumanMessage

Contains the dynamic input data for the current request, such as:

- Title
- Duration
- Genres
- Release year
- Platform delay
- Description

It tells the model **what to analyze**.

---

# Q16. If the LLM consistently flags foreign language films as "anomalous" simply because they are non-English, how do you adjust the prompt to correct this cultural bias?

**Answer:**

Include explicit cultural neutrality instructions in the system prompt.

Example:

> **Note:** Being an international or non-English production is a normal part of our global catalog and **must NOT** be considered a strategic anomaly. Focus only on structural characteristics such as runtime, release delay, unusual genre combinations, and catalog metadata.

This reduces bias and ensures the model evaluates content based on objective metrics rather than language.

---

# Q17. How could you expand this pipeline to pass "Catalog Averages" (e.g., the average duration of a standard documentary) into the prompt to give the LLM better relative context?

**Answer:**

First, compute benchmark statistics using Pandas.

Example:

```python
genre_avg_duration = (
    df.groupby("genres")["duration_num"]
      .mean()
)
```

Before sending the prompt to the LLM:

1. Retrieve the average duration for the title's genre.
2. Pass it as an additional prompt variable.

Example:

```text
Average Documentary Duration: 82 minutes
Current Title Duration: 38 minutes
```

Providing these benchmarks allows the LLM to evaluate anomalies relative to their category instead of relying on general assumptions.

---

# Q18. How do you objectively evaluate the relevance and factual accuracy of the LLM's generated Strategy Briefs across hundreds of titles?

**Answer:**

Implement an automated evaluation framework combining LLM-based assessment and rule-based validation.

Common approaches include:

- **LLM-as-a-Judge** frameworks such as **Ragas** or **DeepEval**
- Automated metadata validation
- Human expert review

Evaluation criteria include:

### Factual Consistency

Verify that metadata mentioned in the strategy brief exactly matches the catalog.

### Groundedness

Ensure every claim is supported by the provided synopsis or metadata.

### Human Review

Randomly sample outputs for periodic review by domain experts to monitor overall quality.

---

# Q19. Explain how "Chain of Thought" prompting improves the LLM's ability to reason about a complex anomaly, such as a 10-minute movie listed in a high-drama category.

**Answer:**

Chain of Thought (CoT) prompting encourages the model to reason through intermediate steps before producing a final answer.

For example, the model may:

1. Compare the runtime against typical drama films.
2. Identify that a 10-minute runtime is highly unusual.
3. Examine the synopsis to determine whether it is:
   - A short film
   - An anthology segment
   - A prologue
   - An experimental production
4. Combine these observations into a well-supported strategic recommendation.

This structured reasoning generally produces more accurate and explainable outputs than directly generating a conclusion.

---

# Q20. When running this pipeline over a large catalog, how do you handle OpenAI API rate limits, token constraints, and potential timeouts during batch processing?

**Answer:**

A scalable production pipeline should incorporate several reliability strategies:

### 1. Asynchronous & Batch Processing

- Use `asyncio` with semaphore-based concurrency control.
- Use the OpenAI Batch API for large non-real-time workloads.

### 2. Exponential Backoff

Retry failed requests using libraries such as **tenacity** to gracefully handle:

- `RateLimitError (429)`
- Timeouts
- Temporary server errors

### 3. Caching

Enable local caching (for example, `SQLiteCache` in LangChain) to avoid repeated API calls for titles that have already been analyzed.

These techniques improve throughput, reduce costs, and make the pipeline more resilient during large-scale processing.