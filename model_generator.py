import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import pickle

with open('career_description_dataset.json', 'r', encoding='utf-8') as file:
    career_data = json.load(file)

# Extract relevant fields
career_names = []
descriptions = []

for entry in career_data:
    career_names.append(entry["career_name"])
    descriptions.append(entry["description"])

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
trained_embeddings = model.encode(descriptions)

data = {
    "model": model,
    "trained_embeddings": trained_embeddings,
    "career_names": career_names,
    "descriptions": descriptions
}

with open('trained_career_model.pickle', 'wb') as f:
    pickle.dump(data, f)

print("Career model training complete and saved.")
