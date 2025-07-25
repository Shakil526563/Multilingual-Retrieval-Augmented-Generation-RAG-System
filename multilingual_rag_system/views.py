from django.http import JsonResponse
from django.urls import reverse


def root_endpoint_list(request):
    """
    Root endpoint that displays all available API endpoints
    """
    base_url = request.build_absolute_uri('/').rstrip('/')
    
    endpoints = [
        {
            "method": "GET",
            "url": f"{base_url}/",
            "name": "Root - List all endpoints",
            "description": "Display all available API endpoints"
        },
        {
            "method": "GET",
            "url": f"{base_url}/api/v1/health/",
            "name": "Health Check",
            "description": "Check system health and initialization status"
        },
        {
            "method": "POST",
            "url": f"{base_url}/api/v1/initialize/",
            "name": "Initialize System",
            "description": "Initialize RAG system with PDF documents",
            "payload": {
                "force_rebuild": "boolean (optional) - Force rebuild vector store"
            }
        },
        {
            "method": "GET",
            "url": f"{base_url}/api/v1/stats/",
            "name": "System Statistics",
            "description": "Get system statistics and status"
        },
        {
            "method": "POST",
            "url": f"{base_url}/api/v1/chat/",
            "name": "Chat Query",
            "description": "Send multilingual queries (English/Bengali)",
            "payload": {
                "query": "string - Your question in English or Bengali",
                "session_id": "string (optional) - Chat session ID"
            }
        },
        {
            "method": "POST",
            "url": f"{base_url}/api/v1/reset/",
            "name": "Reset Conversation",
            "description": "Reset conversation memory",
            "payload": {
                "session_id": "string (optional) - Session ID to reset"
            }
        },
        {
            "method": "GET",
            "url": f"{base_url}/api/v1/sessions/",
            "name": "List Chat Sessions",
            "description": "Get list of all chat sessions"
        },
        {
            "method": "GET",
            "url": f"{base_url}/api/v1/sessions/{'{session_id}'}/",
            "name": "Chat Session Detail",
            "description": "Get details of a specific chat session"
        },
        {
            "method": "GET",
            "url": f"{base_url}/api/v1/metrics/",
            "name": "System Metrics",
            "description": "Get RAG evaluation metrics and performance data"
        },
        {
            "method": "GET",
            "url": f"{base_url}/admin/",
            "name": "Django Admin",
            "description": "Access Django admin interface"
        }
    ]
    
    # Return JSON response only
    return JsonResponse({
        "message": "🌐 Multilingual RAG System - API Endpoints",
        "total_endpoints": len(endpoints),
        "endpoints": endpoints,
        "features": [
            "✅ Multilingual support (English + Bengali)",
            "✅ PDF document processing", 
            "✅ Vector similarity search (FAISS)",
            "✅ LLM integration (Groq)",
            "✅ REST API with Django",
            "✅ Real-time evaluation metrics"
        ],
        "quick_test_commands": [
            f"curl {base_url}/api/v1/health/",
            f'curl -X POST {base_url}/api/v1/chat/ -H "Content-Type: application/json" -d \'{{"query": "What is this document about?"}}\'',
            f"curl {base_url}/api/v1/stats/"
        ]
    }, json_dumps_params={'indent': 2})
