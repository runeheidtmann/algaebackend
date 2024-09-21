from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
import requests
from ..models import Evaluation
from ..serializers import EvaluationSerializer
from rest_framework import status

class EvaluationView(APIView):
    def get(self, request, format=None):
        evaluations = Evaluation.objects.all()
        serializer = EvaluationSerializer(evaluations, many=True, context={'request': request})
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = EvaluationSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class EvaluationDetailView(APIView):
    def get(self, request, id):
        try:
            evaluation = Evaluation.objects.get(id=id)
        except Evaluation.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
        serializer = EvaluationSerializer(evaluation, context={'request': request})
        return Response(serializer.data)
 