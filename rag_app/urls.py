from django.urls import path
from . import views

app_name = 'rag_app'

urlpatterns = [
    # System management
    path('initialize/', views.InitializeSystemView.as_view(), name='initialize_system'),
    path('stats/', views.SystemStatsView.as_view(), name='system_stats'),
    path('health/', views.health_check, name='health_check'),
    path('documents/', views.AvailableDocumentsView.as_view(), name='available_documents'),
    
    # Chat functionality
    path('chat/', views.ChatQueryView.as_view(), name='chat_query'),
    path('reset/', views.reset_conversation, name='reset_conversation'),
    
    # Session management
    path('sessions/', views.ChatSessionListView.as_view(), name='chat_sessions'),
    path('sessions/<str:session_id>/', views.ChatSessionDetailView.as_view(), name='chat_session_detail'),
    
    # Metrics and evaluation
    path('metrics/', views.SystemMetricsView.as_view(), name='system_metrics'),
]
