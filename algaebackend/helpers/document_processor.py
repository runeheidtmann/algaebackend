from ..models import Document
from django.conf import settings
import os

class DocumentProcessor:
    def __init__(self, document_id):
        self.document = self._get_document(document_id)

    def _get_document(self, document_id):
        """
        Retrieve the Document instance based on the document ID.
        """
        try:
            return Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return None

    def process_document_file(self):
        """
        Placeholder for a method to process the document file.
        This could involve reading the file, performing some analysis,
        converting file format, etc.
        """
        
        return False

    def open_and_split_document(self,file_name):
           # Construct the full file path using MEDIA_ROOT
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            print(f"Full file path: {file_path}")
            
            return False
