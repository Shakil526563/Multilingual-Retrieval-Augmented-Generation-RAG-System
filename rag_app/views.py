import os
import uuid
import logging
from datetime import date

from django.shortcuts import render
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import generics

from .models import ChatSession, ChatMessage, DocumentChunk, SystemMetrics
from .serializers import (
    QueryRequestSerializer, QueryResponseSerializer, ChatSessionSerializer,
    ChatMessageSerializer, SystemStatsSerializer, SystemMetricsSerializer
)
from .rag_system import MultilingualRAGSystem

logger = logging.getLogger(__name__)

# Global RAG system instance - initialized lazily
_rag_system = None

def get_rag_system():
    """Get or create the RAG system instance"""
    global _rag_system
    if _rag_system is None:
        _rag_system = MultilingualRAGSystem()
        # Try to auto-initialize if any supported document exists
        try:
            document_path = _find_supported_document()
            if document_path:
                _rag_system.initialize_system(document_path, force_rebuild=False)
                logger.info(f"RAG system auto-initialized successfully with {document_path}")
        except Exception as e:
            logger.warning(f"Failed to auto-initialize RAG system: {e}")
    return _rag_system


def _find_supported_document():
    """Find the first supported document file in the data directory"""
    data_path = settings.PDF_STORAGE_PATH
    supported_extensions = ['.pdf', '.txt', '.csv', '.docx']
    
    if not os.path.exists(data_path):
        logger.warning(f"Data directory not found: {data_path}")
        return None
    
    # Always prefer the specific file if it exists
    preferred_file = 'HSC26-Bangla1st-Paper.txt'
    preferred_path = os.path.join(data_path, preferred_file)
    if os.path.isfile(preferred_path):
        logger.info(f"Found preferred document: {preferred_path}")
        return preferred_path

    # Look for other supported files
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_extensions:
                logger.info(f"Found supported document: {file_path}")
                return file_path
    
    logger.warning("No supported document files found in data directory")
    return None


class InitializeSystemView(APIView):
    """Initialize the RAG system with supported document formats (PDF, TXT, CSV, DOCX)"""
    
    def post(self, request):
        try:
            rag_system = get_rag_system()
            force_rebuild = request.data.get('force_rebuild', False)
            file_name = request.data.get('file_name')  # Optional: specify a particular file
            
            # Find the document to process
            if file_name:
                document_path = os.path.join(settings.PDF_STORAGE_PATH, file_name)
                if not os.path.exists(document_path):
                    return Response(
                        {'error': f'Document file not found at {document_path}'}, 
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                document_path = _find_supported_document()
                if not document_path:
                    return Response(
                        {'error': 'No supported document files (PDF, TXT, CSV, DOCX) found in data directory'}, 
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Check if file format is supported
            _, ext = os.path.splitext(document_path.lower())
            if ext not in ['.pdf', '.txt', '.csv', '.docx']:
                return Response(
                    {'error': f'Unsupported file format: {ext}. Supported formats: PDF, TXT, CSV, DOCX'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            rag_system.initialize_system(document_path, force_rebuild=force_rebuild)
            
            # Update system metrics
            today = date.today()
            metrics, created = SystemMetrics.objects.get_or_create(
                date=today,
                defaults={'system_initialized': True}
            )
            if not created:
                metrics.system_initialized = True
                metrics.save()
            
            return Response({
                'message': 'RAG system initialized successfully',
                'document_path': document_path,
                'file_format': ext.upper(),
                'force_rebuild': force_rebuild
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            return Response(
                {'error': f'Failed to initialize system: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatQueryView(APIView):
    """Handle chat queries to the RAG system"""
    
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            rag_system = get_rag_system()
            if not rag_system.is_initialized:
                return Response(
                    {'error': 'RAG system not initialized. Please initialize first.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            query = serializer.validated_data['query']
            k = serializer.validated_data.get('k', 5)
            session_id = serializer.validated_data.get('session_id')
            
            # Create or get session
            if not session_id:
                session_id = str(uuid.uuid4())
            
            session, created = ChatSession.objects.get_or_create(
                session_id=session_id,
                defaults={'is_active': True}
            )
            
            # Process query through RAG system
            result = rag_system.query(query, k=k)
            
            # Save message to database
            message = ChatMessage.objects.create(
                session=session,
                user_query=query,
                assistant_response=result['response'],
                query_language=result['query_language'],
                response_time=result['evaluation']['response_time'],
                groundedness_score=result['evaluation']['groundedness'],
                relevance_score=result['evaluation']['relevance'],
                retrieved_documents_count=len(result['retrieved_documents'])
            )
            
            # Update system metrics
            self._update_system_metrics(result)
            
            # Prepare response
            response_data = {
                'response': result['response'],
                'session_id': session_id,
                'query_language': result['query_language'],
                'retrieved_documents': result['retrieved_documents'],
                'evaluation': result['evaluation']
            }
            
            response_serializer = QueryResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response(
                {'error': f'Failed to process query: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _update_system_metrics(self, result):
        """Update daily system metrics"""
        today = date.today()
        metrics, created = SystemMetrics.objects.get_or_create(
            date=today,
            defaults={
                'total_queries': 0,
                'avg_response_time': 0.0,
                'avg_groundedness': 0.0,
                'avg_relevance': 0.0,
                'bengali_queries': 0,
                'english_queries': 0
            }
        )
        
        # Update metrics
        total_queries = metrics.total_queries + 1
        metrics.avg_response_time = (
            (metrics.avg_response_time * metrics.total_queries + result['evaluation']['response_time']) 
            / total_queries
        )
        metrics.avg_groundedness = (
            (metrics.avg_groundedness * metrics.total_queries + result['evaluation']['groundedness']) 
            / total_queries
        )
        metrics.avg_relevance = (
            (metrics.avg_relevance * metrics.total_queries + result['evaluation']['relevance']) 
            / total_queries
        )
        metrics.total_queries = total_queries
        
        # Update language-specific metrics
        if result['query_language'] == 'bengali':
            metrics.bengali_queries += 1
        elif result['query_language'] == 'english':
            metrics.english_queries += 1
        
        metrics.save()


class SystemStatsView(APIView):
    """Get system statistics and evaluation metrics"""
    
    def get(self, request):
        try:
            rag_system = get_rag_system()
            stats = rag_system.get_system_stats()
            serializer = SystemStatsSerializer(stats)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return Response(
                {'error': f'Failed to get stats: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AvailableDocumentsView(APIView):
    """List available documents and their formats"""
    
    def get(self, request):
        try:
            data_path = settings.PDF_STORAGE_PATH
            supported_extensions = ['.pdf', '.txt', '.csv', '.docx']
            documents = []
            
            if os.path.exists(data_path):
                for filename in os.listdir(data_path):
                    file_path = os.path.join(data_path, filename)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(filename.lower())
                        if ext in supported_extensions:
                            file_size = os.path.getsize(file_path)
                            documents.append({
                                'filename': filename,
                                'format': ext.upper(),
                                'size_bytes': file_size,
                                'size_mb': round(file_size / (1024 * 1024), 2),
                                'supported': True
                            })
                        else:
                            documents.append({
                                'filename': filename,
                                'format': ext.upper() if ext else 'Unknown',
                                'size_bytes': os.path.getsize(file_path),
                                'supported': False
                            })
            
            return Response({
                'data_directory': data_path,
                'supported_formats': ['PDF', 'TXT', 'CSV', 'DOCX'],
                'total_files': len(documents),
                'supported_files': len([d for d in documents if d['supported']]),
                'documents': documents
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return Response(
                {'error': f'Failed to list documents: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatSessionListView(generics.ListAPIView):
    """List all chat sessions"""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer


class ChatSessionDetailView(generics.RetrieveAPIView):
    """Get details of a specific chat session"""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    lookup_field = 'session_id'


class SystemMetricsView(generics.ListAPIView):
    """Get system performance metrics"""
    queryset = SystemMetrics.objects.all()
    serializer_class = SystemMetricsSerializer


@api_view(['GET'])
def health_check(request):
    """Health check endpoint"""
    try:
        rag_system = get_rag_system()
        is_initialized = rag_system.is_initialized
        
        # Check available documents
        document_path = _find_supported_document()
        has_documents = document_path is not None
        
        if document_path:
            _, ext = os.path.splitext(document_path)
            current_format = ext.upper()
            current_file = os.path.basename(document_path)
        else:
            current_format = None
            current_file = None
            
    except Exception:
        is_initialized = False
        has_documents = False
        current_format = None
        current_file = None
    
    return Response({
        'status': 'healthy',
        'rag_system_initialized': is_initialized,
        'groq_api_configured': bool(settings.GROQ_API_KEY),
        'supported_formats': ['PDF', 'TXT', 'CSV', 'DOCX'],
        'has_documents': has_documents,
        'current_document': current_file,
        'current_format': current_format
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
def reset_conversation(request):
    """Reset conversation memory"""
    try:
        rag_system = get_rag_system()
        session_id = request.data.get('session_id')
        if session_id:
            # Mark session as inactive
            ChatSession.objects.filter(session_id=session_id).update(is_active=False)
        
        # Clear short-term memory if no session specified or global reset
        if not session_id or request.data.get('global_reset', False):
            rag_system.memory.short_term_memory = []
        
        return Response({
            'message': 'Conversation reset successfully',
            'session_id': session_id
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        return Response(
            {'error': f'Failed to reset conversation: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
