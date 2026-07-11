import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file or environment.")
    print("Please create a .env file and add your GEMINI_API_KEY.")
    exit()

print("Authenticating with Gemini API...")
genai.configure(api_key=api_key)

print("\nAvailable Gemini Text Models:")
print("-" * 30)

try:
    models = genai.list_models()
    for m in models:
        # We only care about models that can generate text/content
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model ID: {m.name}")
            print(f"Description: {m.description}")
            print("-" * 30)
except Exception as e:
    print(f"Failed to fetch models: {e}")
