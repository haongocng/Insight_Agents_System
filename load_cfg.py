import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv(".env")

# Set up API keys and environment variables
DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')
DEEPINFRA_API_BASE = "https://api.deepinfra.com/v1/openai"
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
# Get working directory from environment variable
WORKING_DIRECTORY  = os.getenv('WORKING_DIRECTORY ', './storage/')
# Get Conda-related paths from environment variables
CONDA_PATH = os.getenv('CONDA_PATH', '/Users/haong/miniconda3')
CONDA_ENV = os.getenv('CONDA_ENV', 'base')

