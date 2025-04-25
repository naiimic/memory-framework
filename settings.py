import os
from pathlib import Path
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv()

# Paths
DATABASEPATH = BASE_DIR / "generative" / "database"
UNITYPATH = BASE_DIR / "unity"
UNITYBUILDPATH = BASE_DIR / "unity" / "GenAgentJune" / "Builds"
CONFIGPATH = BASE_DIR / "configs"
EXPERIMENTPATH = BASE_DIR / "experiments"

MODELPATH = BASE_DIR / "models"

NONCE = [None, "", " ", "\n", "."]

# Language Models API Keys - get from environment or .env file
OPENAI_APIKEY = os.environ.get("OPENAI_API_KEY")

# Setting environment variables for LangChain, note that the name is different due to convention
if OPENAI_APIKEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_APIKEY

# Huggingface tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    print(f"OpenAI API Key: {'Set' if OPENAI_APIKEY else 'Not Set'}")