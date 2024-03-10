from rest_framework import viewsets
from rest_framework.response import Response
from ..models import Document, DocumentFile
from ..serializers import DocumentSerializer, DocumentFileSerializer
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from ..helpers import DocumentProcessor


class DocumentUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # Pass the request as context to the serializer
        document_serializer = DocumentSerializer(data=request.data, context={'request': request})
        if document_serializer.is_valid():
            document = document_serializer.save()
           
            document_file_serializer = DocumentFileSerializer(data={
                'file': request.data.get('file'),
                'document': document.id
            })
            if not document_file_serializer.is_valid():
                print(document_file_serializer.errors)  # Add this line to debug
                document.delete()  # Clean up if file saving fails
                return Response(document_file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            if document_file_serializer.is_valid():
                doc = document_file_serializer.save()
                print("ok3")
                doc_processor = DocumentProcessor(doc.id)
                doc_processor.process_document_file()
                
                return Response(document_file_serializer.data, status=status.HTTP_201_CREATED)
            else:
                document.delete()  # Clean up if file saving fails
                return Response(document_file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response(document_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
