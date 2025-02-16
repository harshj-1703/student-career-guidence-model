import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import pickle

# Load trained career model
with open('trained_career_model.pickle', 'rb') as f:
    data = pickle.load(f)

model = data["model"]
trained_embeddings = data["trained_embeddings"]
career_names = data["career_names"]
short_descriptions = data["short_descriptions"]
full_descriptions = data["full_descriptions"]

# Query input
question = "I am having interest in coding related stuff what i can do?"

# Encode the question
new_question_embedding = model.encode(question)

# Compute cosine similarity
cosine_similarities = util.dot_score(new_question_embedding, trained_embeddings)
index = np.argmax(cosine_similarities).item()

# Output the best-matching career
print("Career Name:", career_names[index])
print("Short Description:", short_descriptions[index])
print("Full Description:", full_descriptions[index])
