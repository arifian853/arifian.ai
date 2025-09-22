# RAG Chatbot Backend - FastAPI 
### Arifian.AI v1.1

Backend FastAPI untuk sistem chatbot RAG (Retrieval-Augmented Generation) dengan knowledge base management dan embedding search menggunakan SentenceTransformer. Digunakan untuk Arifian.AI pada halaman portofolio di : https://arifian853.vercel.app

## 🚀 Fitur

- **RAG Chatbot**: Sistem chatbot dengan kemampuan retrieval-augmented generation
- **Knowledge Base Management**: CRUD operations untuk mengelola knowledge base
- **Vector Search**: Pencarian similarity menggunakan embedding SentenceTransformer
- **File Upload**: Support upload file TXT, CSV, dan JSON
- **Admin Dashboard**: Web interface untuk mengelola knowledge base
- **MongoDB Integration**: Database NoSQL dengan vector search capabilities
- **Google Gemini Integration**: AI model untuk response generation

## 🛠️ Tech Stack

- **FastAPI**: Modern web framework untuk Python
- **MongoDB**: Database dengan Motor async driver
- **SentenceTransformer**: Model embedding lokal (all-MiniLM-L6-v2)
- **Google Gemini**: AI model untuk text generation
- **Uvicorn**: ASGI server
- **Bootstrap 5**: Frontend UI framework

## 📋 Prerequisites

- Python 3.8+
- MongoDB Atlas account atau MongoDB local
- Google API Key untuk Gemini

## 🔧 Installation

1. **Clone repository**
```bash
git clone <repository-url>
cd backend-fastapi
```

2. **Create dan aktifkan virtual environment**

### Menggunakan pip (venv)
```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Menggunakan uv (recommended - lebih cepat)
```bash
# Install uv jika belum ada
pip install uv

# Buat virtual environment dengan uv
uv venv

# Aktifkan virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**

### Menggunakan pip
```bash
pip install -r requirements.txt
```

### Menggunakan uv (lebih cepat)
```bash
uv pip install -r requirements.txt
```

4. **Setup environment variables**
```bash
cp .env.example .env
```

Edit file `.env` dengan konfigurasi Anda:
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

Server akan berjalan di `http://localhost:8000`

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

Akses admin dashboard di `http://localhost:8000` untuk:

- **Knowledge Base**: View dan manage knowledge items
- **Add Knowledge**: Tambah knowledge item manual
- **Upload Files**: Upload file TXT, CSV, atau JSON

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

1. **Embedding Model**: SentenceTransformer 'all-MiniLM-L6-v2' untuk generate embeddings
2. **Vector Search**: Cosine similarity search pada MongoDB
3. **RAG Pipeline**: Retrieve relevant documents → Generate response dengan Gemini
4. **Async Operations**: Full async/await support dengan Motor driver

## 🔍 Usage Examples

### 1. Chat dengan Knowledge Base
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "Apa itu machine learning?",
    "history": []
})

print(response.json())
```

### 2. Tambah Knowledge Item
```python
import requests

response = requests.post("http://localhost:8000/knowledge", json={
    "title": "Machine Learning Basics",
    "content": "Machine learning adalah subset dari artificial intelligence...",
    "source": "ML Textbook Chapter 1"
})

print(response.json())
```

### 3. Upload File CSV
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
2. **CORS**: Configure allowed origins untuk production
3. **Database**: Use MongoDB Atlas atau dedicated MongoDB instance
4. **Monitoring**: Add logging dan monitoring
5. **Security**: Implement authentication dan rate limiting

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB connection string | Yes |
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Similarity Threshold**: Configurable dalam search function
- **Max Results**: Default 5 documents per search

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push ke branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Pastikan MONGODB_URI benar
   - Check network connectivity
   - Verify MongoDB Atlas whitelist IP

2. **Google API Error**
   - Verify GOOGLE_API_KEY valid
   - Check API quota limits
   - Ensure Gemini API enabled

3. **Embedding Model Download**
   - First run akan download model (~90MB)
   - Pastikan internet connection stable
   - Check disk space available

### Debug Mode

Run dengan debug logging:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## 📞 Support

Untuk pertanyaan atau issues, silakan buat issue di repository atau hubungi maintainer.