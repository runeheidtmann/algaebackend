from django.contrib import admin
from .models import Evaluation,LLM,Document,DocumentFile

admin.site.register([Evaluation,LLM,DocumentFile,Document])