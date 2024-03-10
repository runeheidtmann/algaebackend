from django.db import models
from django.contrib.auth.models import User
from .LLM import LLM

class Evaluation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    LLM = models.ForeignKey(LLM, on_delete=models.SET_NULL, null=True)
    date_posted = models.DateTimeField(auto_now_add=True)
    user_question_raw = models.TextField()
    user_question_enriched = models.TextField()
    LLM_answer = models.TextField()
    user_rating =  models.IntegerField()
    
    def __str__(self):
        return f"{self.user_rating} - {self.LLM}"