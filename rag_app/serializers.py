from rest_framework import serializers
from .models import ChatSession, ChatMessage, DocumentChunk, SystemMetrics


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'user_query', 'assistant_response', 'query_language', 
                 'response_time', 'groundedness_score', 'relevance_score', 
                 'retrieved_documents_count', 'created_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['id', 'session_id', 'created_at', 'updated_at', 'is_active', 'messages']


class QueryRequestSerializer(serializers.Serializer):
    query = serializers.CharField(max_length=2000)
    session_id = serializers.CharField(max_length=100, required=False)
    k = serializers.IntegerField(default=5, min_value=1, max_value=10)


class QueryResponseSerializer(serializers.Serializer):
    response = serializers.CharField()
    session_id = serializers.CharField()
    query_language = serializers.CharField()
    retrieved_documents = serializers.ListField()
    evaluation = serializers.DictField()


class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ['chunk_id', 'content', 'language', 'chunk_size', 
                 'source_document', 'metadata', 'created_at']


class SystemMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SystemMetrics
        fields = ['date', 'total_queries', 'avg_response_time', 'avg_groundedness', 
                 'avg_relevance', 'bengali_queries', 'english_queries', 'system_initialized']


class SystemStatsSerializer(serializers.Serializer):
    is_initialized = serializers.BooleanField()
    total_documents = serializers.IntegerField()
    conversation_history_length = serializers.IntegerField()
    evaluation_summary = serializers.DictField()
    document_languages = serializers.DictField(required=False)
