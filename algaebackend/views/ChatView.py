from rest_framework.views import APIView
from rest_framework.response import Response
import os
import dotenv
dotenv.load_dotenv()
from openai import OpenAI
class ChatAPIView(APIView):

    def post(self, request, *args, **kwargs):
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
        
        return Response(data, 200)