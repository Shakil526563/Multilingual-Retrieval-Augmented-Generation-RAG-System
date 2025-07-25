from django.db import models
from django.contrib.auth.models import User
import json


class ChatSession(models.Model):
    """Model to store chat sessions"""
    session_id = models.CharField(max_length=100, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"Session {self.session_id}"


class ChatMessage(models.Model):
    """Model to store individual chat messages"""
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    user_query = models.TextField()
    assistant_response = models.TextField()
    query_language = models.CharField(max_length=20, default='unknown')
    response_time = models.FloatField(default=0.0)
    groundedness_score = models.FloatField(default=0.0)
    relevance_score = models.FloatField(default=0.0)
    retrieved_documents_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Message in {self.session.session_id} at {self.created_at}"


class DocumentChunk(models.Model):
    """Model to store document chunks and metadata"""
    chunk_id = models.CharField(max_length=100, unique=True)
    content = models.TextField()
    language = models.CharField(max_length=20, default='unknown')
    chunk_size = models.IntegerField(default=0)
    source_document = models.CharField(max_length=255)
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['chunk_id']

    def __str__(self):
        return f"Chunk {self.chunk_id} from {self.source_document}"


class SystemMetrics(models.Model):
    """Model to store system performance metrics"""
    date = models.DateField(auto_now_add=True, unique=True)
    total_queries = models.IntegerField(default=0)
    avg_response_time = models.FloatField(default=0.0)
    avg_groundedness = models.FloatField(default=0.0)
    avg_relevance = models.FloatField(default=0.0)
    bengali_queries = models.IntegerField(default=0)
    english_queries = models.IntegerField(default=0)
    system_initialized = models.BooleanField(default=False)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"Metrics for {self.date}"
