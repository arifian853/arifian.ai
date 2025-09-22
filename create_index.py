from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client.get_database("ai-personal")
knowledge_collection = db.get_collection("knowledge")

# Tambahkan dokumen contoh dengan embedding jika koleksi kosong
if knowledge_collection.count_documents({}) == 0:
    sample_embedding = [0.0] * 384  # Array dengan 384 nilai 0.0
    knowledge_collection.insert_one({
        "title": "Sample Document",
        "content": "This is a sample document to initialize the vector index.",
        "embedding": sample_embedding
    })
    print("Sample document added")

# Create vector search index
try:
    # Hapus indeks yang sudah ada jika ada
    existing_indexes = knowledge_collection.list_indexes()
    for index in existing_indexes:
        if index.get("name") == "vector_index":
            knowledge_collection.drop_index("vector_index")
            print("Existing index dropped")
            break
    
    # Gunakan metode create_index untuk MongoDB Atlas Free Tier
    knowledge_collection.create_index(
        [("embedding", "2dsphere")],
        name="vector_index"
    )
    print("Vector index created successfully")
        
except Exception as e:
    print(f"Error creating vector search index: {str(e)}")
    
client.close()