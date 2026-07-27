import numpy as np
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)
from langchain_ollama import ChatOllama
from sklearn.ensemble import IsolationForest

# =====================================================================
# STEP 1: LOAD & PREPARE DATASET
# =====================================================================
print("📥 Loading Netflix Catalog Dataset...")
df = pd.read_csv('netflix_titles.csv')

# Feature Engineering for Anomaly Detection
df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['added_year'] = pd.to_datetime(df['date_added'], format='mixed').dt.year
df['platform_delay'] = df['added_year'] - df['release_year']

# Filter valid dataset for Machine Learning
features = ['duration_num', 'release_year', 'platform_delay']
valid_df = df.dropna(subset=features).copy()

# =====================================================================
# STEP 2: ARCHITECTURE II - DETERMINISTIC ANOMALY DETECTION (ML)
# =====================================================================
print("\n🌲 Running Isolation Forest Outlier Detection...")
iso_forest = IsolationForest(contamination=0.01, random_state=42)
valid_df['anomaly_score'] = iso_forest.fit_predict(valid_df[features])

# Isolate flagged catalog anomalies (-1 indicates outlier)
anomalies = valid_df[valid_df['anomaly_score'] == -1]
print(f"✅ Detected {len(anomalies)} structural content anomalies!")

# =====================================================================
# STEP 3: INITIALIZE OLLAMA LLM
# =====================================================================
llm = ChatOllama(model="gemma2:2b", temperature=0.2)

# =====================================================================
# STEP 4: ARCHITECTURE II - LLM STRATEGY BRIEF GENERATION
# =====================================================================
print("\n🧠 Generating Strategic Content Brief for Top Anomaly...")

diagnostic_prompt = PromptTemplate.from_template(
    "You are a Senior Content Acquisition Strategist at a major streaming network.\n\n"
    "Our analytics engine flagged the following title as a structural anomaly in our catalog:\n\n"
    "Title: {title}\n"
    "Type: {type}\n"
    "Duration: {duration}\n"
    "Genres: {genres}\n"
    "Release Year: {release_year}\n"
    "Country: {country}\n"
    "Synopsis: {description}\n\n"
    "Analyze these attributes. Is it anomalous because of an extreme runtime, a rare genre combination, "
    "or an unusually long delay between release and platform acquisition?\n"
    "Provide a 3-bullet-point Content Strategy Brief explaining why this stands out from standard programming, "
    "and advise whether acquiring more content like this represents a viable niche audience strategy."
)

strategy_chain = diagnostic_prompt | llm

# Analyze the first flagged anomaly
sample_anomaly = anomalies.iloc[0]
brief_response = strategy_chain.invoke({
    "title": sample_anomaly['title'],
    "type": sample_anomaly['type'],
    "duration": sample_anomaly['duration'],
    "genres": sample_anomaly['listed_in'],
    "release_year": sample_anomaly['release_year'],
    "country": sample_anomaly['country'],
    "description": sample_anomaly['description'],
})

print("\n--- 📋 CONTENT STRATEGY BRIEF ---")
print(brief_response.content)

# =====================================================================
# STEP 5: ARCHITECTURE I - AUTONOMOUS TEXT-TO-PANDAS AGENT
# =====================================================================
print("\n🤖 Initializing Autonomous Content Strategy Agent...")

# Merge anomaly flags into the main DataFrame
df['is_anomaly'] = df.index.isin(anomalies.index)

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    agent_type="zero-shot-react-description",
    allow_dangerous_code=True,
)


# =====================================================================
# STEP 6: INTERACTIVE QUESTION LOOP
# =====================================================================
def start_interactive_agent():
    print("\n" + "=" * 60)
    print("💬 INTERACTIVE DATASET ASSISTANT READY")
    print("Type your question below (or type 'exit' or 'quit' to stop).")
    print("=" * 60 + "\n")

    while True:
        try:
            user_query = input("❓ Enter your question: ").strip()

            if user_query.lower() in ['exit', 'quit']:
                print("\n👋 Exiting assistant. Goodbye!")
                break

            if not user_query:
                continue

            print("\n🔍 Querying dataset...")
            response = agent.invoke(user_query)

            print("\n💡 Response:")
            if isinstance(response, dict) and 'output' in response:
                print(response['output'])
            else:
                print(response)

            print("\n" + "-" * 60)

        except Exception as e:
            print(f"\n❌ An error occurred while executing query: {e}\n")


# Launch the interactive session
if __name__ == "__main__":
    start_interactive_agent()