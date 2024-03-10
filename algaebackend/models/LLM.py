from django.db import models

class LLM(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(null=True)
    
    
    def __str__(self):
        return self.name