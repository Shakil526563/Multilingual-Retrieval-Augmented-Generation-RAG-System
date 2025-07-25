# Multilingual RAG System - Setup & Run Guide

## 🚀 Quick Start Guide

This guide provides step-by-step instructions to set up and run the Multilingual RAG System.

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Python 3.8+ installed
- ✅ Git (optional, for cloning)
- ✅ PDF document in the `data/` folder
- ✅ Internet connection (for package installation)

## 🛠️ Installation Steps

### Step 1: Navigate to Project Directory
```bash
cd "e:\assesment of tenminutes"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Set Up Environment Variables
Create a `.env` file with your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 6: Run Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

## 🌐 Running the System

### Method 1: Standard Django Server

1. **Start the Django Server:**
```bash
python manage.py runserver
```

2. **Initialize the RAG System (PowerShell):**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/initialize/" -Method POST -ContentType "application/json" -Body "{}"
```

3. **Check System Health:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health/" -Method GET
```

4. **Test Chat Functionality:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat/" -Method POST -ContentType "application/json" -Body '{"query": "What is this document about?"}'
```

### Method 2: Using cURL (Alternative)

1. **Initialize System:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/initialize/ \
  -H "Content-Type: application/json" \
  -d "{}"
```

2. **Health Check:**
```bash
curl http://127.0.0.1:8000/api/v1/health/
```

3. **Send Chat Query:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

## 🔗 Available Endpoints

Once the server is running, access these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root - List all endpoints |
| GET | `/api/v1/health/` | System health check |
| POST | `/api/v1/initialize/` | Initialize RAG system |
| GET | `/api/v1/stats/` | System statistics |
| POST | `/api/v1/chat/` | Send multilingual queries |
| POST | `/api/v1/reset/` | Reset conversation |
| GET | `/api/v1/sessions/` | List chat sessions |
| GET | `/api/v1/metrics/` | Performance metrics |
| GET | `/admin/` | Django admin interface |

## 🧪 Testing the System

### Complete Test Sequence (PowerShell)

```powershell
# 1. Start server (in separate terminal)
python manage.py runserver

# 2. Initialize system
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/initialize/" -Method POST -ContentType "application/json" -Body "{}"

# 3. Verify health
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health/" -Method GET

# 4. Test English query
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat/" -Method POST -ContentType "application/json" -Body '{"query": "What is this document about?"}'

# 5. Test Bengali query
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat/" -Method POST -ContentType "application/json" -Body '{"query": "এই ডকুমেন্টটি কী নিয়ে?"}'

# 6. Check system statistics
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/stats/" -Method GET

# 7. View all endpoints
Invoke-RestMethod -Uri "http://127.0.0.1:8000/" -Method GET
```

## 📊 Expected Responses

### Successful Initialization
```json
{
  "message": "RAG system initialized successfully",
  "pdf_path": "E:\\assesment of tenminutes\\data\\Shakil Rana.pdf",
  "force_rebuild": false
}
```

### Health Check (After Initialization)
```json
{
  "status": "healthy",
  "rag_system_initialized": true,
  "groq_api_configured": true
}
```

### Chat Response Example
```json
{
  "response": "The documents appear to be about Shakil Rana, a Machine Learning Engineer...",
  "session_id": "497d30ea-8af8-491c-8252-c989fdb105fb",
  "query_language": "english",
  "retrieved_documents": [...],
  "evaluation": {
    "groundedness": 0.5,
    "relevance": 0.12,
    "response_time": 6.0
  }
}
```

## 🎯 Features

- ✅ **Multilingual Support**: English and Bengali (বাংলা)
- ✅ **PDF Processing**: Automatic document processing
- ✅ **Vector Search**: FAISS-based similarity search
- ✅ **LLM Integration**: Groq API with Llama3-8B
- ✅ **REST API**: Django REST Framework
- ✅ **Evaluation Metrics**: Real-time performance tracking
- ✅ **Session Management**: Conversation memory
- ✅ **Admin Interface**: Django admin panel

## 🔧 Troubleshooting

### Common Issues

1. **"RAG system not initialized" Error:**
   - Run the initialization command before using chat
   - Ensure PDF file exists in `data/` folder

2. **Server won't start:**
   - Check if port 8000 is available
   - Verify all dependencies are installed

3. **Groq API errors:**
   - Verify your API key in `.env` file
   - Check internet connection

4. **Import errors:**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

### Reset System
If you need to reset the system:
```powershell
# Stop server (Ctrl+C)
# Clear vector database
Remove-Item -Recurse -Force vector_db
# Restart server and reinitialize
```

## 📝 Development Notes

- **Database**: SQLite (included)
- **Vector Store**: FAISS (auto-created)
- **Embeddings**: HuggingFace sentence-transformers
- **LLM**: Groq Llama3-8B-8192
- **Framework**: Django 4.2.7 + DRF

## 🎉 Success Indicators

System is ready when you see:
- ✅ Server running on `http://127.0.0.1:8000`
- ✅ Health check returns `"rag_system_initialized": true`
- ✅ Chat queries return responses
- ✅ Both English and Bengali queries work

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all prerequisites are met
3. Ensure API keys are properly configured
4. Check server logs for detailed error messages

---

**🌐 Your Multilingual RAG System is ready to handle questions in English and Bengali!**
