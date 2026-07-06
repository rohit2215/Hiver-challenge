import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a local .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("CRITICAL: OPENAI_API_KEY is missing from your .env file.")

# Initialize the global OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define global model config
MODEL_NAME = "gpt-4o-mini"
