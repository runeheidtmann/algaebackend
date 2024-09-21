from rest_framework import serializers
from rest_framework.serializers import Serializer, FileField
from .models import Evaluation, LLM, Document, DocumentFile
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'
        extra_kwargs = {'password': {'write_only': True}}

class EvaluationSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = Evaluation
        fields = '__all__'
        extra_kwargs = {'user': {'read_only': True}}

    def create(self, validated_data):
        user = self.context['request'].user
        return Evaluation.objects.create(user=user, **validated_data)

class LLMSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLM
        fields = ['name', 'description']

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password']

    def create(self, validated_data):
        user = User.objects.create_user(
            validated_data['username'],
            validated_data['email'],
            validated_data['password']
        )
        user.first_name = validated_data.get('first_name', '')
        user.last_name = validated_data.get('last_name', '')
        user.save()
        return user

class DocumentFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentFile
        fields = ['id', 'file', 'document']

class DocumentSerializer(serializers.ModelSerializer):
    document_files = DocumentFileSerializer(many=True, read_only=True, source='documentfile_set')

    class Meta:
        model = Document
        fields = ['id', 'title', 'document_files']

    def create(self, validated_data):
        user = self.context.get('request').user
        document = Document.objects.create(user_id=user.id, **validated_data)
        return document