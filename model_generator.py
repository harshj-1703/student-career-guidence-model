import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import pickle

# Load career data JSON
with open('career_dataset.json', 'r', encoding='utf-8') as file:
    career_data = json.load(file)

# Extract relevant fields
career_names = []
short_descriptions = []
full_descriptions = []

for entry in career_data:
    career_names.append(entry["career_name"])
    short_descriptions.append(entry["short_description"])
    full_descriptions.append(entry["full_description"])

# Combine short and full descriptions for better embeddings
combined_descriptions = [f"{name}: {short} {full}" for name, short, full in zip(career_names, short_descriptions, full_descriptions)]

# Load and train the model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
trained_embeddings = model.encode(combined_descriptions)

# Save trained model and data
data = {
    "model": model,
    "trained_embeddings": trained_embeddings,
    "career_names": career_names,
    "short_descriptions": short_descriptions,
    "full_descriptions": full_descriptions
}

with open('trained_career_model.pickle', 'wb') as f:
    pickle.dump(data, f)

print("Career model training complete and saved.")
