from django.db import models
from .Chat import ChatMessage


class ResponseMetrics(models.Model):
    """
    Tracks response time metrics for RAG pipeline stages.
    Linked to assistant ChatMessage for analytics.
    """
    chat_message = models.OneToOneField(
        ChatMessage, 
        on_delete=models.CASCADE, 
        related_name='response_metrics'
    )
    
    # Timing for each pipeline stage (in milliseconds)
    session_handling_ms = models.IntegerField(default=0, help_text="Time to get/create session and load history")
    vector_search_ms = models.IntegerField(default=0, help_text="Time for Pinecone vector search")
    entity_extraction_ms = models.IntegerField(default=0, help_text="Time for LLM entity extraction")
    graph_expansion_ms = models.IntegerField(default=0, help_text="Time for Neo4j graph traversal")
    llm_generation_ms = models.IntegerField(default=0, help_text="Time for final answer generation")
    total_ms = models.IntegerField(default=0, help_text="Total request processing time")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Response metrics"
    
    def __str__(self):
        return f"Metrics for message {self.chat_message_id} - {self.total_ms}ms"
    
    @classmethod
    def create_from_timings(cls, chat_message, timings: dict):
        """
        Create ResponseMetrics from a timings dictionary.
        
        Args:
            chat_message: The ChatMessage instance to link to
            timings: Dict with keys like 'vector_search_ms', 'total_ms', etc.
        """
        return cls.objects.create(
            chat_message=chat_message,
            session_handling_ms=timings.get('session_handling_ms', 0),
            vector_search_ms=timings.get('vector_search_ms', 0),
            entity_extraction_ms=timings.get('entity_extraction_ms', 0),
            graph_expansion_ms=timings.get('graph_expansion_ms', 0),
            llm_generation_ms=timings.get('llm_generation_ms', 0),
            total_ms=timings.get('total_ms', 0),
        )
