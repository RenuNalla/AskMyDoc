"""
Standalone MongoDB connection test — isolates auth issues from the rest
of the pipeline. Run this on its own to debug the connection separately.

Usage:
    python test_connection.py
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.environ["MONGODB_URI"]
print(f"Using URI (password hidden): {uri.split('@')[0].split('://')[0]}://***@{uri.split('@')[-1] if '@' in uri else uri}")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    print("SUCCESS: connected and authenticated.")
    print("Databases visible:", client.list_database_names())
except Exception as e:
    print("FAILED:", type(e).__name__, "-", e)