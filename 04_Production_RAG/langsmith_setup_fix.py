# LangSmith Setup Fix
# Run this before any other LangChain imports

import os
from dotenv import load_dotenv
import getpass

# 1. Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# 2. Set LangSmith environment variables BEFORE any LangChain imports
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "AIE7-RAG-Project"

# 3. Check for API key
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key:
    print("LANGSMITH_API_KEY not found in .env file")
    langsmith_api_key = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# 4. Verify setup
print("=== LangSmith Setup Verification ===")
print(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"LANGSMITH_API_KEY: {'✅ Present' if os.getenv('LANGSMITH_API_KEY') else '❌ Missing'}")

# 5. Test connection
try:
    from langsmith import Client
    client = Client()
    # Try to access a simple endpoint
    client.list_projects(limit=1)
    print("✅ LangSmith connection successful!")
except Exception as e:
    print(f"❌ LangSmith connection failed: {e}")
    print("Please check your API key and internet connection")

print("=== Setup Complete ===")
print("Now you can import and use LangChain components") 