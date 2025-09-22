from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
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
    If you don't know the answer based on the provided context, say so or you can just say that you can't say that because it was too personal or a secret, but on't too harsh when answering, be a good person, humble and never overshare. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English. 
    
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

# Create admin.html template
admin_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Admin Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .container { max-width: 1200px; }
        .knowledge-item { margin-bottom: 15px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .nav-tabs { margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">RAG Knowledge Base Admin</h1>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="knowledge-tab" data-bs-toggle="tab" data-bs-target="#knowledge" type="button" role="tab">Knowledge Base</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="add-tab" data-bs-toggle="tab" data-bs-target="#add" type="button" role="tab">Add Knowledge</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="upload-tab" data-bs-toggle="tab" data-bs-target="#upload" type="button" role="tab">Upload Files</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Knowledge Base Tab -->
            <div class="tab-pane fade show active" id="knowledge" role="tabpanel">
                <h3>Knowledge Items</h3>
                <div id="knowledge-list" class="mt-4">
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Add Knowledge Tab -->
            <div class="tab-pane fade" id="add" role="tabpanel">
                <h3>Add New Knowledge</h3>
                <form id="add-knowledge-form">
                    <div class="form-group">
                        <label for="title">Title</label>
                        <input type="text" class="form-control" id="title" name="title" required>
                    </div>
                    <div class="form-group">
                        <label for="content">Content</label>
                        <textarea class="form-control" id="content" name="content" rows="6" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="source">Source (optional)</label>
                        <input type="text" class="form-control" id="source" name="source">
                    </div>
                    <button type="submit" class="btn btn-primary">Add Knowledge</button>
                </form>
            </div>
            
            <!-- Upload Files Tab -->
            <div class="tab-pane fade" id="upload" role="tabpanel">
                <h3>Upload Files</h3>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h5>Upload Text File (.txt)</h5>
                    </div>
                    <div class="card-body">
                        <form id="upload-txt-form" enctype="multipart/form-data">
                            <div class="form-group">
                                <label for="txt-file">Text File</label>
                                <input type="file" class="form-control" id="txt-file" name="file" accept=".txt" required>
                            </div>
                            <div class="form-group">
                                <label for="txt-title">Title</label>
                                <input type="text" class="form-control" id="txt-title" name="title" required>
                            </div>
                            <div class="form-group">
                                <label for="txt-source">Source (optional)</label>
                                <input type="text" class="form-control" id="txt-source" name="source">
                            </div>
                            <button type="submit" class="btn btn-primary">Upload</button>
                        </form>
                    </div>
                </div>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h5>Upload CSV File (.csv)</h5>
                    </div>
                    <div class="card-body">
                        <form id="upload-csv-form" enctype="multipart/form-data">
                            <div class="form-group">
                                <label for="csv-file">CSV File</label>
                                <input type="file" class="form-control" id="csv-file" name="file" accept=".csv" required>
                            </div>
                            <div class="form-group">
                                <label for="title-column">Title Column</label>
                                <input type="text" class="form-control" id="title-column" name="title_column" required>
                            </div>
                            <div class="form-group">
                                <label for="content-column">Content Column</label>
                                <input type="text" class="form-control" id="content-column" name="content_column" required>
                            </div>
                            <button type="submit" class="btn btn-primary">Upload</button>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5>Upload JSON File (.json)</h5>
                    </div>
                    <div class="card-body">
                        <form id="upload-json-form" enctype="multipart/form-data">
                            <div class="form-group">
                                <label for="json-file">JSON File</label>
                                <input type="file" class="form-control" id="json-file" name="file" accept=".json" required>
                            </div>
                            <div class="form-group">
                                <label for="title-field">Title Field</label>
                                <input type="text" class="form-control" id="title-field" name="title_field" required>
                            </div>
                            <div class="form-group">
                                <label for="content-field">Content Field</label>
                                <input type="text" class="form-control" id="content-field" name="content_field" required>
                            </div>
                            <button type="submit" class="btn btn-primary">Upload</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Load knowledge items
        async function loadKnowledge() {
            try {
                const response = await fetch('/knowledge');
                const data = await response.json();
                
                const knowledgeList = document.getElementById('knowledge-list');
                knowledgeList.innerHTML = '';
                
                if (data.length === 0) {
                    knowledgeList.innerHTML = '<div class="alert alert-info">No knowledge items found.</div>';
                    return;
                }
                
                data.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'knowledge-item';
                    div.innerHTML = `
                        <h4>${item.title}</h4>
                        <p><strong>Source:</strong> ${item.source || 'N/A'}</p>
                        <p>${item.content.substring(0, 200)}${item.content.length > 200 ? '...' : ''}</p>
                        <button class="btn btn-sm btn-danger delete-btn" data-id="${item._id}">Delete</button>
                    `;
                    knowledgeList.appendChild(div);
                });
                
                // Add event listeners to delete buttons
                document.querySelectorAll('.delete-btn').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        if (confirm('Are you sure you want to delete this item?')) {
                            const id = e.target.getAttribute('data-id');
                            try {
                                const response = await fetch(`/knowledge/${id}`, {
                                    method: 'DELETE'
                                });
                                
                                if (response.ok) {
                                    alert('Knowledge deleted successfully');
                                    loadKnowledge();
                                } else {
                                    const error = await response.json();
                                    alert(`Error: ${error.detail}`);
                                }
                            } catch (error) {
                                alert(`Error: ${error.message}`);
                            }
                        }
                    });
                });
            } catch (error) {
                console.error('Error loading knowledge:', error);
                document.getElementById('knowledge-list').innerHTML = `
                    <div class="alert alert-danger">Error loading knowledge: ${error.message}</div>
                `;
            }
        }
        
        // Add knowledge form
        document.getElementById('add-knowledge-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                title: document.getElementById('title').value,
                content: document.getElementById('content').value,
                source: document.getElementById('source').value || null
            };
            
            try {
                const response = await fetch('/knowledge', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                if (response.ok) {
                    alert('Knowledge added successfully');
                    document.getElementById('add-knowledge-form').reset();
                    loadKnowledge();
                    // Switch to knowledge tab
                    document.getElementById('knowledge-tab').click();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
        
        // Upload TXT form
        document.getElementById('upload-txt-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('file', document.getElementById('txt-file').files[0]);
            formData.append('title', document.getElementById('txt-title').value);
            formData.append('source', document.getElementById('txt-source').value || '');
            
            try {
                const response = await fetch('/upload-txt', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    alert('Text file uploaded successfully');
                    document.getElementById('upload-txt-form').reset();
                    loadKnowledge();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
        
        // Upload CSV form
        document.getElementById('upload-csv-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('file', document.getElementById('csv-file').files[0]);
            formData.append('title_column', document.getElementById('title-column').value);
            formData.append('content_column', document.getElementById('content-column').value);
            
            try {
                const response = await fetch('/upload-csv', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    alert(`CSV file processed: ${result.message}`);
                    document.getElementById('upload-csv-form').reset();
                    loadKnowledge();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
        
        // Upload JSON form
        document.getElementById('upload-json-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('file', document.getElementById('json-file').files[0]);
            formData.append('title_field', document.getElementById('title-field').value);
            formData.append('content_field', document.getElementById('content-field').value);
            
            try {
                const response = await fetch('/upload-json', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    alert(`JSON file processed: ${result.message}`);
                    document.getElementById('upload-json-form').reset();
                    loadKnowledge();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
        
        // Load knowledge on page load
        document.addEventListener('DOMContentLoaded', loadKnowledge);
    </script>
</body>
</html>
"""

# Create templates directory and admin.html file
os.makedirs("templates", exist_ok=True)
with open("templates/admin.html", "w") as f:
    f.write(admin_html)

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)