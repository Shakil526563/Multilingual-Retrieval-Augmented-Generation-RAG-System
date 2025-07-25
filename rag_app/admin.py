from django.contrib import admin
from .models import ChatSession, ChatMessage, DocumentChunk, SystemMetrics


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'created_at', 'updated_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['session_id', 'user__username']
    readonly_fields = ['created_at', 'updated_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'query_language', 'response_time', 'groundedness_score', 'relevance_score', 'created_at']
    list_filter = ['query_language', 'created_at']
    search_fields = ['user_query', 'assistant_response', 'session__session_id']
    readonly_fields = ['created_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('session')


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ['chunk_id', 'language', 'chunk_size', 'source_document', 'created_at']
    list_filter = ['language', 'source_document', 'created_at']
    search_fields = ['chunk_id', 'content', 'source_document']
    readonly_fields = ['created_at']


@admin.register(SystemMetrics)
class SystemMetricsAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_queries', 'avg_response_time', 'avg_groundedness', 'avg_relevance', 'system_initialized']
    list_filter = ['date', 'system_initialized']
    readonly_fields = ['date']
