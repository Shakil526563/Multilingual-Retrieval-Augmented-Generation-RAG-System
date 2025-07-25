# Multilingual RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system capable of understanding and responding to queries in both English and Bengali. Built with Django REST Framework, LangChain, FAISS, and Groq API.

## 🌟 Features

### Core Functionality
- **Multilingual Support**: Handles queries in English and Bengali (বাংলা)
- **Document Processing**: Extracts and processes text from PDF and others documents
- **Intelligent Text Chunking**: Optimized chunking for better retrieval accuracy
- **Vector Search**: FAISS-based semantic similarity search
- **Memory Management**: Short-term (conversation) and long-term (document corpus) memory
- **REST API**: Comprehensive API for all system interactions

### Advanced Features
- **Language Detection**: Automatic detection of query language
- **Bengali Text Processing**: Specialized preprocessing for Bengali text
- **Conversation Context**: Maintains conversation history for better responses
- **RAG Evaluation**: Built-in metrics for groundedness and relevance
- **System Monitoring**: Performance metrics and health checks

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


### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
```bash
# .env file is already created with:
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=your_django_secret_key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
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

## Sample queries and outputs (Bangla & English)
**Sample Test Case:**


https://github.com/user-attachments/assets/2231e255-0671-435a-ae0f-039b635d6a89

**Sample Test Case: with English** 


https://github.com/user-attachments/assets/6ced551e-af7b-4f25-9f3a-a4ce5cbcf049



## 🔧 System Components

### 1. PDF Processor
- Extracts text from PDF files
- Handles Bengali and English text
- Intelligent text cleaning and normalization
- Document chunking with overlap

### 2. Vector Store (FAISS)
- High-performance similarity search
- Persistent storage
- Batch processing support
- Configurable similarity thresholds

### 3. LLM Integration (Groq)
- Fast inference with Llama3-8B
- Optimized prompts for multilingual responses
- Context-aware generation
- Error handling and fallbacks

### 4. Memory Management
- **Short-term**: Recent conversation history
- **Long-term**: Document corpus in vector database
- Configurable memory limits
- Context preservation across queries

## 🏗️ Architecture
<img width="378" height="536" alt="image" src="https://github.com/user-attachments/assets/bc775801-6a88-46ff-a007-f4ae57f29d83" />



## 📖 API Documentation

### Base URL: `http://localhost:8000/api/v1/`

### Endpoints

#### 1. System Management

**Health Check**
```http
GET /health/
```
Response:
```json
{
    "status": "healthy",
    "rag_system_initialized": true,
    "groq_api_configured": true
}
```

**Initialize System**
```http
POST /initialize/
Content-Type: application/json

{
    "force_rebuild": false
}
```

**System Statistics**
```http
GET /stats/
```
Response:
```json
{
    "is_initialized": true,
    "total_documents": 150,
    "conversation_history_length": 5,
    "evaluation_summary": {
        "avg_groundedness": 0.85,
        "avg_relevance": 0.92,
        "avg_response_time": 2.5
    },
    "document_languages": {
        "bengali": 80,
        "english": 70
    }
}
```

#### 2. Chat Interface

**Send Query**
```http
POST /chat/
Content-Type: application/json

{
    "query": "What is this document about?",
    "session_id": "optional-session-id",
    "k": 5
}
```
Response:
```json
{
    "response": "This document discusses...",
    "session_id": "generated-session-id",
    "query_language": "english",
    "retrieved_documents": [
        {
            "content": "Document chunk content...",
            "similarity_score": 0.95,
            "metadata": {
                "chunk_id": 1,
                "language": "english"
            }
        }
    ],
    "evaluation": {
        "groundedness": 0.85,
        "relevance": 0.92,
        "response_time": 2.1
    }
}
```

**Reset Conversation**
```http
POST /reset/
Content-Type: application/json

{
    "session_id": "optional-session-id",
    "global_reset": false
}
```

#### 3. Session Management

**List Sessions**
```http
GET /sessions/
```

**Get Session Details**
```http
GET /sessions/{session_id}/
```

#### 4. Metrics

**System Metrics**
```http
GET /metrics/
```

- **Multilingual Input**: Type in English or Bengali
- **Real-time Chat**: Instant responses with typing indicators
- **System Status**: Live system health monitoring
- **Performance Metrics**: Response time, relevance scores
- **Responsive Design**: Works on desktop and mobile



### 5. Evaluation System
- **Groundedness**: Response support in retrieved context
- **Relevance**: Document relevance to query
- **Performance**: Response time monitoring
- **Language Analysis**: Query language distribution

## 📊 Performance Metrics

The system tracks various metrics:

- **Response Time**: Average query processing time
- **Groundedness Score**: How well responses are supported by retrieved documents
- **Relevance Score**: Semantic similarity between queries and retrieved documents
- **Language Distribution**: Usage patterns across languages
- **System Health**: API availability and initialization status

## 🛠️ Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=django_secret_key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

### System Settings
```python
# In settings.py
PDF_STORAGE_PATH = BASE_DIR / 'data'
VECTOR_DB_PATH = BASE_DIR / 'vector_db'

# Vector store configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SIMILARITY_THRESHOLD = 0.7
MAX_RETRIEVED_DOCS = 5

# Memory configuration
MAX_SHORT_TERM_MEMORY = 10
```

## 📈 Evaluation Results

The system includes comprehensive evaluation metrics:

### Groundedness Evaluation
- Measures how well responses are supported by retrieved context
- Uses keyword overlap and semantic analysis
- Typical scores: 0.7-0.9 for well-grounded responses

### Relevance Evaluation
- Semantic similarity between queries and retrieved documents
- Uses cosine similarity of embeddings
- Typical scores: 0.8-0.95 for relevant retrievals



### Common Issues

**1. System Not Initializing**
```bash
# Check PDF file exists
ls data/
# Force rebuild
python manage.py initialize_rag --force-rebuild
```

# Multilingual RAG System: Technical Q&A

## 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

**Text Extraction:**
The system uses **PyMuPDF** (fitz) for PDF text extraction. This library is chosen for its robust support for Unicode, including Bengali script, and its ability to extract text with font and layout information.

**Formatting Challenges:**
Yes, I did. Extracting Bengali text from the PDF was particularly challenging due to inconsistent formatting and encoding issues. Because Bengla PDFs often have font encoding issues, missing ligatures, or broken word boundaries. I spent almost two days resolving this.
After extensive research, I found that converting the PDF to a txt file provided more control over the content. To ensure high-quality extraction, my code applies multiple methods—standard, font-aware, and layout-based extraction
Additionally, post-processing steps were implemented to clean up whitespace, punctuation, and common Bengali/English formatting artifacts, making the text suitable for question answering.

## 2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

I used a character-based chunking method using **RecursiveCharacterTextSplitter** from LangChain.
Each chunk has 1500 characters, and there is a 200-character overlap between chunks.

**Why it works well:**
This method keeps each chunk at a good size — not too small and not too big — so the meaning stays clear.
The overlap helps the system remember the connection between one chunk and the next.
This makes it easier for the system to understand the text and give better answers when searching or matching with questions.
## 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

I used the model called **sentence-transformers/all-MiniLM-L6-v2** from HuggingFace.

**Why I chose it:**
This model is fast, efficient and gives good results.
It works well for both English and Bengali text.
It’s popular for tasks like search and question-answering, and it fits well with FAISS for storing and finding similar vectors.

**How it understands meaning:**
The model turns each sentence or paragraph into a vector — a list of numbers that represent its meaning.
Then, we can compare these vectors to find which texts are most similar to a user’s question.
## 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

I use **cosine similarity** to compare the user's question and the stored text chunks.
The data is stored and searched using FAISS (Facebook AI Similarity Search).

**Why this method:**
Cosine similarity helps measure how similar two pieces of text are by comparing their meaning (as vectors). It looks at the “angle” between the question and the document vectors.
FAISS is a very fast and powerful tool that helps search through a large number of stored text chunks quickly and accurately.
## 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

I use the **same embedded model** to turn both the user’s question and the document chunks into vectors, so they are in the same format and meaning space.
I split the documents carefully and added a bit of overlap between chunks so that important context is not lost.
 
 **What if the question is vague or unclear?**
If the question doesn’t give enough information, the system might return less accurate or unrelated chunks.
In such cases, the system is designed to say something like “Not enough information found” instead of giving a wrong answer.
## 6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?
**Relevance:**
Yes, the evaluation results show high relevance and accuracy on the test set. The results are accurate when the question is clear and the document is well-written.
If the question is vague or the document has poor formatting, the results might be less accurate.

**How to improve:**
Use a Bengali-specific or multilingual embedding model to better understand Bengali queries.
Improve chunking by using sentence-based chunks or adapting chunk size based on the content.
After retrieving chunks, apply extra filtering or post-processing to better match the user's intent.


## 📝 Note:
This project uses only free models and tools. Due to limitations in handling large files, some pages were removed to ensure better accuracy and performance in answering queries.If OpenAI or other premium models were used, the system could process larger files faster and generate more accurate responses.


## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **LangChain**: For RAG framework
- **FAISS**: For vector similarity search
- **Groq**: For fast LLM inference
- **HuggingFace**: For embeddings
- **Django**: For web framework
- **Sentence Transformers**: For text embeddings

---

**Built with ❤️ for multilingual document understanding**
