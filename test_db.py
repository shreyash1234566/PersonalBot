import os
import sys
from pymongo import MongoClient
import certifi
from urllib.parse import quote_plus  # <--- NEW: Tool to fix passwords

# ---------------------------------------------------------
# 1. ENTER YOUR CREDENTIALS HERE
# ---------------------------------------------------------
# Replace 'admin' if your username is different
username = quote_plus("admin") 

# Your password with the '@' symbol is now safe because of quote_plus
password = quote_plus("shreyash1@bB") 

# 2. YOUR CORRECT CLUSTER ADDRESS (Check MongoDB Atlas if this is wrong)
# Based on your previous screenshots, your cluster ID is likely "cluster0.qz37998"
cluster_url = "cluster0.qz37998.mongodb.net"

# 3. BUILD THE SECURE URI
MONGO_URI = f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "test"

def test_connection():
    print(f"🔄 Attempting to connect to: {cluster_url}")
    print(f"🔑 Using Username: {username}")
    
    try:
        # Connect
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        
        # Ping
        client.admin.command("ping")
        print("\n✅ SUCCESS! Connection established.")

        db = client.get_database(DB_NAME)
        
        # Test Write
        res = db.debug_logs.insert_one({"status": "working", "source": "local_script_fixed"})
        print(f"📝 Write Test Successful! Inserted ID: {res.inserted_id}")

        # Cleanup
        db.debug_logs.delete_one({"_id": res.inserted_id})
        print("🧹 Cleanup done. Your database is working perfectly.")

    except Exception as e:
        print("\n❌ CONNECTION FAILED")
        print("---------------------")
        print(f"Error: {e}")
        print("---------------------")
        print("Double check your Username (is it 'admin'?) and Cluster URL.")

if __name__ == "__main__":
    test_connection()