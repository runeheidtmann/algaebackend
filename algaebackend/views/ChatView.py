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
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
dotenv.load_dotenv()
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

class ChatAPIView(APIView):

    def post(self, request, *args, **kwargs):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'YourAPIKey')
        PINECONE_ENV = os.getenv('PINECONE_ENV', 'gcp-starter') 
        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  
            environment=PINECONE_ENV  
        )
        index_name = "algaeopenai" 

        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

        # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`

        #docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)


        #already existing index:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        index = pinecone.Index(index_name)
        text_field = "text"  # the metadata field that contains our text

        # initialize the vector store object
        vectorstore = Pinecone(
            index, embeddings, text_field
        )
        llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        query = request.data.get('question')
      
        docs = vectorstore.similarity_search(query, k=3)
        answer = chain.run(input_documents=docs, question=query)
        data = {
            "question": query,
            "docs": docs,
            "answer": answer, 
        }
        return Response(data, 200)