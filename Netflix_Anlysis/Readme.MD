# 📊 Netflix Catalog Analytics & Technical Assessment

---

## 📌 Executive Summary
This document outlines a comprehensive exploratory data analysis (EDA) and technical assessment conducted on the **Netflix Catalog Dataset**. The evaluation is structured into two main tracks:
1. **Theoretical & Conceptual Analysis:** 15 foundational questions covering core Python data science concepts, memory allocation, indexing mechanics, statistical distributions, and time-series logic.
2. **Practical Data Pipeline:** 15 hands-on coding solutions applied directly to `netflix_titles.csv` to clean, transform, aggregate, and visualize catalog data efficiently.

---

## 🧠 Part 1: Theoretical & Conceptual Formulations

### ⚡ Memory & Execution Optimization
* **`object` vs. `category` Allocation:** `object` columns store full Python string objects or pointers per row, duplicating memory for repeating strings. `category` columns construct an internal dictionary array of unique strings and map each row to lightweight integer pointers (e.g., `int8`, `int16`), drastically lowering RAM usage.
  > ⚠️ **Degradation Edge Case:** In high-cardinality columns (e.g., UUIDs, timestamps) where almost every string is unique, the computational overhead of managing, hashing, and searching a massive dictionary index outweighs memory benefits, slowing down sorting and grouping.
* **NumPy Universal Functions (`ufuncs`):** `ufuncs` execute element-wise operations by running compiled, low-level C code loops over flat, continuous blocks of homogeneous memory, utilizing SIMD (Single Instruction, Multiple Data) CPU architectures. Standard Python list comprehensions rely on dynamic pointer tracking, requiring the interpreter to repeatedly inspect object wrappers, check dynamic type flags, and resolve method references per element.
* **Vectorization vs. `.apply()`:** Pandas `.str` accessors call optimized C-level string operations over series blocks. Custom regex lambda functions passed via `.apply()` trigger a slow, row-by-row Python interpreter loop.

### 🔍 Indexing & Data Structures
* **Indexing Mechanics (`loc[]` vs. `iloc[]`):** `.loc[]` is strictly label-based (inclusive of endpoint slice ranges). `.iloc[]` is strictly integer position-based (0-indexed layout coordinates, exclusive of endpoints).
  > 💡 **Edge Case:** On non-consecutive integer indices (e.g., `[3, 7, 9]`), calling `df.loc[0]` triggers a `KeyError` or returns an incorrect row. Calling `df.iloc[0]` safely extracts the true physical first row.
* **DataFrame Joins (`pd.merge()`):** Defines `inner` (exact matches), `outer` (retains all rows with null fills), `left` (keeps all left rows and matches right), and `right` (keeps all right rows and matches left). Missing key matches in a `left` join populate all corresponding right-side fields with `NaN` or `None`.
* **Split-Apply-Combine & GroupBy Lazy Evaluation:** The paradigm divides data into sub-populations (**Split**), computes functions per group (**Apply**), and stitches outputs back together (**Combine**). Executing `df.groupby('key')` creates a lightweight `DataFrameGroupBy` object mapping row index addresses without duplicating data; actual splitting and computation are deferred until an aggregation method (e.g., `.sum()`, `.agg()`) is invoked.
* **GroupBy `.transform()` vs. `.agg()`:** `.transform()` applies operations per group and broadcasts calculated summary values back onto the row indices of the original dataset, matching the exact input shape. Conversely, `.agg()` collapses individual rows to return exactly one summary row per unique group category[cite: 6].
* **Unnesting via `.explode()`:** Transforms list-like cell elements into separate, independent rows while preserving and duplicating original index labels across the expanded output[cite: 6].
* **Time-Series Resampling:** Converts date/time frequencies (downsampling or upsampling) using a time-aware index engine that fills chronological gaps and supports forward/backward interpolation (`ffill`/`bfill`), unlike standard `.groupby()` which treats dates as discrete text buckets[cite: 6].

### 📈 Statistical Analysis & Visual Paradigms
* **Kernel Density Estimation (KDE) & Bandwidth:** Calculates a smooth probability density curve by placing a small Gaussian kernel over every data point and summing them together[cite: 6]. The **bandwidth** controls kernel width: small bandwidths overfit data into a jagged, noisy curve; large bandwidths oversmooth the distribution, flattening out real structural peaks[cite: 6].
* **Violinplot vs. Boxplot:** Boxplots display static summary statistics (quartiles, median, outliers) while hiding actual distribution shapes[cite: 6]. Violinplots wrap those summary metrics inside a continuous KDE curve, revealing hidden multimodal peaks, skewness, and data clustering[cite: 6].
* **Multicollinearity:** A condition where independent features are highly correlated, introducing redundant information[cite: 6]. It is identified by computing `df.corr(method='pearson')` and passing the numeric matrix to `sns.heatmap(annot=True, cmap='coolwarm', vmin=-1, vmax=1)` to target highly correlated intersections ($\vert{}r\vert{} \ge 0.80$) to drop prior to model training[cite: 6].
* **Matplotlib Interfaces:** `pyplot` implicitly tracks active figures and subplots automatically[cite: 6]. The Object-Oriented interface (`fig, ax`) explicitly instantiates container figures and plot grids, allowing absolute control over individual elements without canvas contamination[cite: 6].
* **Interquartile Range (IQR):** Derived mathematically as $IQR = Q_3 - Q_1$, capturing the central 50% of the distribution[cite: 6]. Visually represented as the central box container in a boxplot[cite: 6].

---

## 🛠️ Part 2: Practical Pipeline & Methodology
```text
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Memory-Efficient Import │ ──> │ Just-In-Time Pipeline   │ ──> │ Relational Normalization│
│  (Categorical Casting)  │     │  (Vectorized Parsing)   │     │   (.str.split & explode)│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```


### 1. Data Cleaning & Feature Engineering
* **⚡ Memory-Efficient Ingestion:** Mapped repetitive textual fields (`type`, `country`, `rating`, `listed_in`) directly to categorical dtypes at load time to compress initial RAM footprint.
* **📊 Missing Value Auditing:** Computed column-wise null percentages using `df.isnull().mean().sort_values(ascending=False) * 100`.
* **🚩 Boolean Flagging:** Generated binary indicator flags using `np.where(df['director'].notnull(), True, False)`.
* **⏱️ Temporal & String Extraction:** Parsed string dates via `pd.to_datetime()` to extract `month_added`, and stripped non-alphanumeric noise from title strings using Regex `.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)`.
* **📅 Generational Bucketing:** Binned release years into discrete eras (*20th Century*, *2000s*, *2010s+*) using `np.select()`.

### 2. Transformations & Aggregations
* **🔗 List Normalization:** Exploded comma-separated cast lists via `.assign(cast=df['cast'].str.split(', ')).explode('cast')`.
* **🧩 Cross-Tabulation:** Built cross-tabular matrix counts of content ratings vs. types using `df.pivot_table(index='rating', columns='type', values='show_id', aggfunc='count', fill_value=0)`.
* **⛓️ Method Chaining:** Formulated query chains to isolate top 10 movie directors using `.dropna()`, `.query()`, `.groupby()`, `.count()`, and `.nlargest()`.

### 3. Statistical Profiling & Visualizations
* **📐 Outlier Detection:** Isolated 450 movie duration outliers falling outside mathematical IQR bounds:
  $$IQR = Q_3 - Q_1$$
  $$\text{Valid Range} = [Q_1 - 1.5 \times IQR, \; Q_3 + 1.5 \times IQR]$$
* **🎨 Multi-Panel Graphics:** Rendered sorted rating countplots with Seaborn, alongside dual-histogram subplots (`fig, axes = plt.subplots(1, 2)`) comparing release years and numerical movie runtimes.


### 💾 Exported Output Dataset: `netflix_titles_cleaned.csv`

The data processing pipeline produces a newly structured, production-ready dataset saved as **`netflix_titles_cleaned.csv`**.

---

#### 🌟 What is New?
The cleaned dataset introduces **5 newly engineered feature columns**:

* 🚩 **`is_director_known`** *(Boolean)*: Evaluates to `True` if director metadata exists, and `False` if missing.
* 🗓️ **`month_added`** *(String)*: Extracted text name of the month when the title was added to the platform (parsed from `date_added`).
* ⏱️ **`duration_numeric`** *(Float)*: Isolated numeric runtime values in minutes (extracted from string duration descriptions).
* 🔤 **`title_clean`** *(String)*: Cleaned title strings stripped of non-alphanumeric special characters and punctuation.
* 📅 **`era_category`** *(Categorical)*: Segmented release eras into discrete generational groups (*20th Century*, *2000s*, *2010s+*).

---

#### 🔄 What is Transformed?

##### 🏷️ Renamed Schema Headers
* `listed_in` $\longrightarrow$ **`genres`**
* `date_added` $\longrightarrow$ **`platform_addition_date`**

##### ⚡ Structural & Performance Enhancements
* ⏱️ **Parsed Datatypes:** `platform_addition_date` was converted from raw text into a structured `datetime64` object.
* ⚡ **RAM Footprint Optimization:** High-repetition fields (`type`, `country`, `rating`, `genres`) were explicitly mapped to memory-efficient `category` pointer types during ingestion.