from rest_framework.views import APIView
from rest_framework.response import Response
import os
from rest_framework import status
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

class RagChatAPIView(APIView):

    def post(self, request, *args, **kwargs):
        try:
            query=request.data.get('question')
            
            load_dotenv()
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
            index_name= os.getenv("PINECONE_INDEX_NAME")
            
            #init LLM and Vectorstore
            model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
            parser = StrOutputParser()
            embeddings = OpenAIEmbeddings()
            vectorstore = PineconeVectorStore.from_existing_index(index_name,embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            
            #Setup context retriever
            setup = RunnableParallel(context=retriever, question=RunnablePassthrough())
            context = setup.invoke(query)
            
            #Build prompt and chain it all together
            prompt = self.buildPrompt()
            prompt_text = prompt.format(context=context,question=query)
            
            #build question chain
            chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | parser
                )
            
            #invoke answer
            answer = chain.invoke(query)
        
            #Build respons to be handled in the frontend
            data = {
                "question": query,
                "docs": context,
                "answer": answer,
                "prompt": prompt_text, 
            }

            #return with data and 200 response
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": "An error occurred while processing your request"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def buildPrompt(self):
        template = """
        You are an algae research assistant. Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt