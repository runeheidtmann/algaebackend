from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import ConversationChain
import os
from algaebackend import serializers
import dotenv
dotenv.load_dotenv()
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


class ChatAPIView(APIView):

    def get(self, request, *args, **kwargs):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)