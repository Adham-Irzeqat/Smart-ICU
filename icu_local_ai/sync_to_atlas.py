from pymongo import MongoClient
import streamlit as st

# Local MongoDB
local_client = MongoClient("mongodb://localhost:27017/")
local_db = local_client["icu_database"]

# Atlas MongoDB
atlas_client = MongoClient("")
# atlas_client = MongoClient(st.secrets["MONGO_URI"])
atlas_db = atlas_client["icu_database"]

for collection_name in local_db.list_collection_names():
    local_col = local_db[collection_name]
    atlas_col = atlas_db[collection_name]
    
    # Find documents that are not yet in Atlas
    for doc in local_col.find():
        if not atlas_col.find_one({"_id": doc["_id"]}):
            atlas_col.insert_one(doc)

print("✅ Sync complete! Only new records copied.")


