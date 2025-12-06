from django.contrib import admin
from .models import Evaluation,LLM,Document,DocumentFile,ChatSession,ChatMessage,ResponseMetrics

admin.site.register([Evaluation,LLM,DocumentFile,Document,ChatSession,ChatMessage])


@admin.register(ResponseMetrics)
class ResponseMetricsAdmin(admin.ModelAdmin):
    list_display = ['id', 'chat_message', 'total_ms', 'vector_search_ms', 'llm_generation_ms', 'created_at']
    list_filter = ['created_at']
    ordering = ['-created_at']
    readonly_fields = ['chat_message', 'session_handling_ms', 'vector_search_ms', 
                       'entity_extraction_ms', 'graph_expansion_ms', 'llm_generation_ms', 
                       'total_ms', 'created_at']