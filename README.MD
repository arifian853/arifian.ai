<h3 align="center"> Arifian.AI v1.1 </h3>

<p align="center"> FastAPI backend for RAG (Retrieval-Augmented Generation) chatbot system with knowledge base management and embedding search using SentenceTransformer. Used for Personal my-own-AI-persona Chatbot Arifian.AI on portfolio page : https://arifian853.vercel.app </p>

<div align="center">
    <img src="https://img.shields.io/badge/FastAPI-009485?style=for-the-badge&logo=fastapi&logoColor=white">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white">
    <img src="https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white">
    <img src="https://img.shields.io/badge/Uvicorn-FF6C37?style=for-the-badge&logo=uvicorn&logoColor=white">
    <img src="https://img.shields.io/badge/Bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white">
</div>

## 🚀 Features

- **RAG Chatbot**: Chatbot system with retrieval-augmented generation capabilities
- **Knowledge Base Management**: CRUD operations for managing knowledge base
- **Vector Search**: Similarity search using SentenceTransformer embeddings
- **File Upload**: Support for uploading TXT, CSV, and JSON files
- **Admin Dashboard**: Web interface for managing knowledge base
- **MongoDB Integration**: NoSQL database with vector search capabilities
- **Google Gemini Integration**: AI model for response generation

## 🛠️ Tech Stack

- **FastAPI**: Modern web framework for Python
- **MongoDB**: Database with Motor async driver
- **SentenceTransformer**: Local embedding model (all-MiniLM-L6-v2)
- **Google Gemini**: AI model for text generation
- **Uvicorn**: ASGI server
- **Bootstrap 5**: Frontend UI framework

## 📋 Prerequisites

- Python 3.8+
- MongoDB Atlas account or local MongoDB
- Google API Key for Gemini

## 🔧 Installation

1. **Clone repository**
```bash
git clone <repository-url>
cd backend-fastapi
```

2. **Create and activate virtual environment**

### Using pip (venv)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Using uv (recommended - faster)
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment with uv
uv venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**

### Using pip
```bash
pip install -r requirements.txt
```

### Using uv (faster)
```bash
uv pip install -r requirements.txt
```

4. **Setup environment variables**
```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Initialize MongoDB vector index**
```bash
python create_index.py
```

6. **Run the application**
```bash
python app.py
```

Server will run at `http://localhost:8000`

## 📚 API Documentation

### Chat Endpoint

**POST** `/chat`
```json
{
  "message": "Your question here",
  "history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

Response:
```json
{
  "response": "AI generated response",
  "sources": [
    {
      "title": "Source document title",
      "content": "Relevant content snippet",
      "source": "Document source"
    }
  ]
}
```

### Knowledge Management

**GET** `/knowledge` - List all knowledge items

**POST** `/knowledge` - Create new knowledge item
```json
{
  "title": "Document title",
  "content": "Document content",
  "source": "Optional source",
  "metadata": {}
}
```

**GET** `/knowledge/{knowledge_id}` - Get specific knowledge item

**PUT** `/knowledge/{knowledge_id}` - Update knowledge item

**DELETE** `/knowledge/{knowledge_id}` - Delete knowledge item

### File Upload

**POST** `/upload-txt` - Upload text file
- Form data: `file`, `title`, `source` (optional)

**POST** `/upload-csv` - Upload CSV file
- Form data: `file`, `title_column`, `content_column`

**POST** `/upload-json` - Upload JSON file
- Form data: `file`, `title_field`, `content_field`

## 🖥️ Admin Dashboard

Access admin dashboard at `http://localhost:8000` for:

- **Knowledge Base**: View and manage knowledge items
- **Add Knowledge**: Add knowledge items manually
- **Upload Files**: Upload TXT, CSV, or JSON files

## 🏗️ Architecture

```
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── create_index.py     # MongoDB vector index setup
├── .env.example       # Environment variables template
├── templates/         # Jinja2 templates
│   └── admin.html     # Admin dashboard
└── static/           # Static files
```

### Key Components

1. **Embedding Model**: SentenceTransformer 'all-MiniLM-L6-v2' for generating embeddings
2. **Vector Search**: Cosine similarity search on MongoDB
3. **RAG Pipeline**: Retrieve relevant documents → Generate response with Gemini
4. **Async Operations**: Full async/await support with Motor driver

## 🔍 Usage Examples

### 1. Chat with Knowledge Base
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "What is machine learning?",
    "history": []
})

print(response.json())
```

### 2. Add Knowledge Item
```python
import requests

response = requests.post("http://localhost:8000/knowledge", json={
    "title": "Machine Learning Basics",
    "content": "Machine learning is a subset of artificial intelligence...",
    "source": "ML Textbook Chapter 1"
})

print(response.json())
```

### 3. Upload CSV File
```python
import requests

files = {'file': open('data.csv', 'rb')}
data = {
    'title_column': 'title',
    'content_column': 'description'
}

response = requests.post("http://localhost:8000/upload-csv", files=files, data=data)
print(response.json())
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t rag-backend .

# Run container
docker run -p 8000:8000 --env-file .env rag-backend
```

### Production Considerations

1. **Environment Variables**: Set proper production values
2. **CORS**: Configure allowed origins for production
3. **Database**: Use MongoDB Atlas or dedicated MongoDB instance
4. **Monitoring**: Add logging and monitoring
5. **Security**: Implement authentication and rate limiting

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB connection string | Yes |
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Similarity Threshold**: Configurable in search function
- **Max Results**: Default 5 documents per search

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Ensure MONGODB_URI is correct
   - Check network connectivity
   - Verify MongoDB Atlas IP whitelist

2. **Google API Error**
   - Verify GOOGLE_API_KEY is valid
   - Check API quota limits
   - Ensure Gemini API is enabled

3. **Embedding Model Download**
   - First run will download model (~90MB)
   - Ensure stable internet connection
   - Check available disk space

### Debug Mode

Run with debug logging:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## 📞 Support

For questions or issues, please create an issue in the repository or contact the maintainer.