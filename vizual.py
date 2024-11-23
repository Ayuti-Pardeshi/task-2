# Import necessary libraries
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from transformers import pipeline
import os

# Suppress the Symlink Warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load cleaned JSON data
with open("cleaned_chat_data.json", "r") as file:
    cleaned_data = json.load(file)

# Convert cleaned JSON to a list of messages for analysis
messages = [entry["message"] for entry in cleaned_data]

# Specify the model explicitly to avoid warning
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to split long messages into chunks
def split_message(message, max_length=512):
    """Split messages into chunks that fit the model's token limit."""
    return [message[i:i + max_length] for i in range(0, len(message), max_length)]

# Split long messages into smaller chunks
split_messages = []
original_message_indices = []  # Track which chunks belong to which original message
for idx, message in enumerate(messages):
    chunks = split_message(message)
    split_messages.extend(chunks)
    original_message_indices.extend([idx] * len(chunks))  # Record the mapping

# Apply sentiment analysis to all message chunks
chunk_sentiments = [sentiment_analyzer(chunk)[0]['label'] for chunk in split_messages]

# Aggregate sentiments back to the original messages
message_sentiments = []
for idx in range(len(messages)):
    # Collect all sentiment results for the chunks of this message
    chunk_results = [chunk_sentiments[i] for i in range(len(original_message_indices)) if original_message_indices[i] == idx]
    
    # Aggregate sentiment (e.g., majority vote, or fallback to first if tie)
    aggregated_sentiment = Counter(chunk_results).most_common(1)[0][0]
    message_sentiments.append(aggregated_sentiment)

# Count sentiment categories
sentiment_categories = Counter(message_sentiments)

# Visualization 1: Top 10 User Queries (Bar Chart)
query_counts = Counter(messages)
top_queries = query_counts.most_common(10)
queries, counts = zip(*top_queries)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(counts), y=list(queries), hue=list(queries), palette="Blues_d", legend=False)
plt.title("Top 10 User Queries", fontsize=16)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Query", fontsize=12)
plt.tight_layout()
plt.show()

# Visualization 2: Sentiment Distribution (Bar Chart)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(sentiment_categories.keys()), y=list(sentiment_categories.values()), hue=list(sentiment_categories.keys()), palette="RdYlGn", legend=False)
plt.title("Sentiment Distribution", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Visualization 3: Sentiment Distribution as a Doughnut Pie Chart
labels = sentiment_categories.keys()
sizes = sentiment_categories.values()

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#32CD32", "#FF6347", "#D3D3D3"], 
        wedgeprops={'edgecolor': 'black'})
plt.title("Sentiment Distribution by Category", fontsize=16)
plt.axis("equal")
plt.tight_layout()
plt.show()

# Visualization 4: Sentiment Trends Over Time (Line Chart)
df = pd.DataFrame({"message": messages, "sentiment": message_sentiments, "date": pd.date_range(start="2023-01-01", periods=len(messages), freq="D")})

# Group by date and sentiment
sentiment_trends = df.groupby(["date", "sentiment"]).size().unstack()

# Plot sentiment trends over time
sentiment_trends.plot(kind="line", figsize=(12, 6), linewidth=2)
plt.title("Sentiment Trends Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Queries", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
