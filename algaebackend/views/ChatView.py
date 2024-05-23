from rest_framework.views import APIView
from rest_framework.response import Response
import os
from rest_framework import status
import dotenv
dotenv.load_dotenv()
from openai import OpenAI

class ChatAPIView(APIView):

    def post(self, request, *args, **kwargs):
        try:
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
            client = OpenAI(api_key=OPENAI_API_KEY)
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant in the field of algae, you are an expert and knowledgeable in all algae research."},
                {"role": "user", "content": request.data.get('question')}
            ]
            )
            data = {
                "question": request.data.get('question'),
                "answer": completion.choices[0].message.content, 
            }
            return Response(data, status=status.HTTP_200_OK) 
        
        except Exception as e:
            return Response({"error": "An error occurred while processing your request"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)