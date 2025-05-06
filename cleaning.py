import json
import re

# Load JSON data
with open("chat_history.json", "r") as file:
    chat_data = json.load(file)
#steps Explained all below
# Step 1: Extract human (user) messages
cleaned_conversations = []
for client_id, messages in chat_data.items():
    for message in messages:
        if "human" in message:  # Only include human messages
            cleaned_conversations.append({"client_id": client_id, "message": message["human"]})

# Step 2: Normalize text
def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = text.strip()  # Remove extra whitespace
    return text

for conversation in cleaned_conversations:
    conversation["message"] = normalize_text(conversation["message"])

# Step 3: Remove duplicates
seen_messages = set()
deduplicated_conversations = []
for conversation in cleaned_conversations:
    if conversation["message"] not in seen_messages:
        deduplicated_conversations.append(conversation)
        seen_messages.add(conversation["message"])

# Step 4: Remove irrelevant or short messages
filtered_conversations = [
    conversation for conversation in deduplicated_conversations if len(conversation["message"]) > 2
]

# Save cleaned data back to JSON
with open("cleaned_chat_data.json", "w") as outfile:
    json.dump(filtered_conversations, outfile, indent=4)

# Print the first few cleaned conversations
print(filtered_conversations[:5])
