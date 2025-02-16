import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HUGGING_FACE_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {token}"}

INPUT_JSON_FILE = "career_list.json"
OUTPUT_JSON_FILE = "career_description_dataset.json"

DELIMITER = "### OUTPUT START ###"

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def query_mistral(prompt, max_length=500, temperature=0.7):
    """Send request to the Mistral model."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": max_length, "temperature": temperature}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

career_list = load_json(INPUT_JSON_FILE)

career_data = []

for career in career_list:
    prompt = (
        f"Provide a detailed explanation of the career '{career}'.\n\n"
        f"- Explain the responsibilities and tasks involved.\n"
        f"- Mention key industries where this career is relevant.\n"
        f"- Describe important skills required for success.\n\n"
        f"{DELIMITER}"
    )
    
    response = query_mistral(prompt)

    if isinstance(response, list) and "generated_text" in response[0]:
        generated_text = response[0]["generated_text"]

        description = generated_text.split(DELIMITER, 1)[-1].strip()
    else:
        description = ""
        
    description = description.replace("OUTPUT END", "").strip()

    print(f"Processed: {career}")

    career_data.append({
        "career_name": career,
        "description": description
    })

save_json(career_data, OUTPUT_JSON_FILE)

print(f"Updated JSON saved as '{OUTPUT_JSON_FILE}'")
