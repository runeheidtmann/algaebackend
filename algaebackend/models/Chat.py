from django.db import models
from django.contrib.auth.models import User

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    title = models.CharField(max_length=200, blank=True)  # Auto-generated or user-set
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.title or f'Session {self.id}'}"

class ChatMessage(models.Model):
    MESSAGE_TYPES = [
        ('user', 'User Question'),
        ('assistant', 'Assistant Answer'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Store RAG-specific metadata
    metadata = models.JSONField(null=True, blank=True)  # Store entities, docs, prompt, etc.
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.session.id} - {self.message_type} - {self.created_at}"
