from ..models import Document, DocumentFile
from langchain.document_loaders import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.conf import settings
import os
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma


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
        if self.document:
            document_file = DocumentFile.objects.filter(document=self.document).first()
            if document_file:
                try:
                    #Take file and split it into many text chuncks
                    texts = self.open_and_split_document(str(document_file.file))
                    print("1")
                    #take textchunks and embed them.
                    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    # load it into Chroma
                    print("2")
                    Chroma.from_documents(texts, embedding_function, persist_directory="./chroma_db")
                              
                except:
                    print(f"{document_file.file} can't be loaded correctly")
            return True
        return False

    def open_and_split_document(self,file_name):
           # Construct the full file path using MEDIA_ROOT
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            print(f"Full file path: {file_path}")
            
            # Assuming PyPDFLoader is a custom or third-party class for handling PDFs
            loader = PyPDFLoader(file_path)
            #list of pages
            data = loader.load_and_split()
            list_of_pages = [x for x in data]
            # We'll split our data into chunks around 2056 characters each with a 256 character overlap.
            # 2056 characters is completely arbitrary, around a half page. 
            # - Play around wither larger and smaller chunks to maybe improve results.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2056, chunk_overlap=256)
            return text_splitter.split_documents(list_of_pages)
        
    
    # Add more processing methods as needed
