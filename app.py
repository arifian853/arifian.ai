from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import csv
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import motor.motor_asyncio
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import io
import asyncio

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates for admin UI
templates = Jinja2Templates(directory="templates")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI, server_api=ServerApi('1'))
db = client.get_database("ai-personal")
knowledge_collection = db.get_collection("knowledge")

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Good balance of speed and quality

# Initialize Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = []

class KnowledgeItem(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

# Helper functions
async def generate_embedding(text: str):
    """Generate embedding for text using SentenceTransformer"""
    embedding = model.encode(text)
    return embedding.tolist()

async def add_knowledge(knowledge: KnowledgeItem):
    """Add knowledge to database with embedding"""
    embedding = await generate_embedding(knowledge.content)
    document = {
        "title": knowledge.title,
        "content": knowledge.content,
        "source": knowledge.source,
        "metadata": knowledge.metadata,
        "embedding": embedding
    }
    result = await knowledge_collection.insert_one(document)
    return result.inserted_id

async def search_similar_documents(query: str, limit: int = 5):
    """Search for similar documents by computing similarity in the application"""
    query_embedding = await generate_embedding(query)
    
    # Fetch all documents (consider pagination for large collections)
    documents = []
    async for doc in knowledge_collection.find({}):
        documents.append(doc)
    
    # Calculate cosine similarity for each document
    results_with_scores = []
    for doc in documents:
        # Skip documents without embeddings
        if "embedding" not in doc or not doc["embedding"]:
            continue
            
        # Calculate cosine similarity
        doc_embedding = doc["embedding"]
        similarity = cosine_similarity(query_embedding, doc_embedding)
        
        # Add document with similarity score
        doc_with_score = {
            "_id": doc["_id"],
            "title": doc["title"],
            "content": doc["content"],
            "source": doc.get("source"),
            "metadata": doc.get("metadata", {}),
            "score": similarity
        }
        results_with_scores.append(doc_with_score)
    
    # Sort by similarity score (highest first) and limit results
    results_with_scores.sort(key=lambda x: x["score"], reverse=True)
    top_results = results_with_scores[:limit]
    
    return top_results

# Tambahkan fungsi helper untuk menghitung cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    # Convert to numpy arrays if they aren't already
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Prevent division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    return float(similarity)

async def process_chat(message: str, history: List[Dict[str, str]]):
    """Process chat message with RAG approach"""
    # Search for relevant context
    relevant_docs = await search_similar_documents(message)
    
    # Format context for the model
    context = "\n\n".join([f"Title: {doc['title']}\nContent: {doc['content']}" for doc in relevant_docs])
    
    # Format conversation history
    formatted_history = []
    for entry in history:
        if 'user' in entry:
            formatted_history.append({"role": "user", "parts": [entry['user']]})
        if 'assistant' in entry:
            formatted_history.append({"role": "model", "parts": [entry['assistant']]})
    
    # Create Gemini model
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Start conversation
    chat = model.start_chat(history=formatted_history)
    
    # Generate response with context
    prompt = f"""
    Based on the following information, please answer the user's question.
    If you don't know the answer based on the provided context, say so or you can just say that you can't say that because it was too personal or a secret, but on't too harsh when answering, be a good person, humble and never overshare. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English. If the user is asking about my profile, give the link in markdown so user can just click it.
    
    Context:
    {context}
    
    User question: {message}
    """
    
    response = await asyncio.to_thread(lambda: chat.send_message(prompt))
    
    return {
        "response": response.text,
        "sources": [{"title": doc["title"], "content": doc["content"][:200] + "...", "source": doc["source"]} for doc in relevant_docs]
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def admin_ui(request: Request):
    """Admin UI for knowledge management"""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for frontend integration"""
    try:
        result = await process_chat(request.message, request.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge", response_model=List[Dict[str, Any]])
async def list_knowledge():
    """List all knowledge items"""
    items = []
    async for doc in knowledge_collection.find({}, {"embedding": 0}):
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    return items

@app.post("/knowledge")
async def create_knowledge(knowledge: KnowledgeItem):
    """Add a new knowledge item"""
    result = await add_knowledge(knowledge)
    return {"id": str(result), "message": "Knowledge added successfully"}

@app.post("/upload-txt")
async def upload_txt(
    file: UploadFile = File(...),
    title: str = Form(...),
    source: Optional[str] = Form(None)
):
    """Upload a text file as knowledge"""
    content = await file.read()
    text = content.decode("utf-8")
    
    knowledge = KnowledgeItem(
        title=title,
        content=text,
        source=source or file.filename,
        metadata={"file_type": "txt", "filename": file.filename}
    )
    
    result = await add_knowledge(knowledge)
    return {"id": str(result), "message": "Text file processed successfully"}

@app.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    title_column: str = Form(...),
    content_column: str = Form(...)
):
    """Upload a CSV file as multiple knowledge items"""
    content = await file.read()
    text = content.decode("utf-8")
    
    csv_file = io.StringIO(text)
    csv_reader = csv.DictReader(csv_file)
    
    results = []
    for row in csv_reader:
        if title_column in row and content_column in row:
            knowledge = KnowledgeItem(
                title=row[title_column],
                content=row[content_column],
                source=file.filename,
                metadata={"file_type": "csv", "filename": file.filename, "row_data": row}
            )
            result = await add_knowledge(knowledge)
            results.append(str(result))
    
    return {"ids": results, "message": f"Processed {len(results)} items from CSV"}

@app.post("/upload-json")
async def upload_json(
    file: UploadFile = File(...),
    title_field: str = Form(...),
    content_field: str = Form(...)
):
    """Upload a JSON file as multiple knowledge items"""
    content = await file.read()
    text = content.decode("utf-8")
    
    data = json.loads(text)
    
    results = []
    if isinstance(data, list):
        for item in data:
            if title_field in item and content_field in item:
                knowledge = KnowledgeItem(
                    title=item[title_field],
                    content=item[content_field],
                    source=file.filename,
                    metadata={"file_type": "json", "filename": file.filename, "item_data": item}
                )
                result = await add_knowledge(knowledge)
                results.append(str(result))
    
    return {"ids": results, "message": f"Processed {len(results)} items from JSON"}

@app.delete("/knowledge/{knowledge_id}")
async def delete_knowledge(knowledge_id: str):
    """Delete a knowledge item"""
    from bson import ObjectId
    result = await knowledge_collection.delete_one({"_id": ObjectId(knowledge_id)})
    if result.deleted_count:
        return {"message": "Knowledge deleted successfully"}
    raise HTTPException(status_code=404, detail="Knowledge not found")

@app.get("/knowledge/{knowledge_id}")
async def get_knowledge(knowledge_id: str):
    """Get a specific knowledge item"""
    from bson import ObjectId
    doc = await knowledge_collection.find_one({"_id": ObjectId(knowledge_id)}, {"embedding": 0})
    if doc:
        doc["_id"] = str(doc["_id"])
        return doc
    raise HTTPException(status_code=404, detail="Knowledge not found")

@app.put("/knowledge/{knowledge_id}")
async def update_knowledge(knowledge_id: str, knowledge: KnowledgeItem):
    """Update a knowledge item"""
    from bson import ObjectId
    
    # Generate new embedding for updated content
    embedding = await generate_embedding(knowledge.content)
    
    # Update document with new data and embedding
    update_data = {
        "title": knowledge.title,
        "content": knowledge.content,
        "source": knowledge.source,
        "embedding": embedding
    }
    
    result = await knowledge_collection.update_one(
        {"_id": ObjectId(knowledge_id)},
        {"$set": update_data}
    )
    
    if result.modified_count:
        return {"message": "Knowledge updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Knowledge item not found")

# Create directories for templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add PUT route for updating knowledge
@app.put("/knowledge/{knowledge_id}")
async def update_knowledge(knowledge_id: str, knowledge: KnowledgeItem):
    try:
        # Update the knowledge item
        result = knowledge_collection.update_one(
            {"_id": ObjectId(knowledge_id)},
            {"$set": {
                "title": knowledge.title,
                "content": knowledge.content,
                "source": knowledge.source
            }}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        # Get the updated item
        updated_item = knowledge_collection.find_one({"_id": ObjectId(knowledge_id)})
        if updated_item:
            updated_item["_id"] = str(updated_item["_id"])
            
            # Update embeddings
            embedding = embedding_model.encode(updated_item["content"])
            embedding_collection.update_one(
                {"knowledge_id": knowledge_id},
                {"$set": {"embedding": embedding.tolist()}}
            )
            
            return updated_item
        else:
            raise HTTPException(status_code=404, detail="Knowledge item not found after update")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)