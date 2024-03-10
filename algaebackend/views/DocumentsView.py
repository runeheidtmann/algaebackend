from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from ..models import Document
from ..serializers import DocumentSerializer

class DocumentsView(generics.ListAPIView):
    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Filter documents by the logged-in user
        user = self.request.user
        return Document.objects.filter(user=user)