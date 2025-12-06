from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from ..models import ChatSession, ChatMessage
from ..serializers import ChatSessionSerializer, ChatSessionListSerializer, ChatMessageSerializer
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

class ChatSessionListCreateView(generics.ListCreateAPIView):
    """
    GET: List all chat sessions for authenticated user
    POST: Create a new chat session
    """
    permission_classes = [IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'GET':
            return ChatSessionListSerializer
        return ChatSessionSerializer
    
    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user, is_active=True)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class ChatSessionDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    GET: Retrieve a specific chat session with full message history
    PATCH/PUT: Update chat session (e.g., title)
    DELETE: Soft delete chat session (set is_active=False)
    """
    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer
    
    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user)
    
    def retrieve(self, request, *args, **kwargs):
        response = super().retrieve(request, *args, **kwargs)
        print("\n" + "="*80)
        print("CHAT SESSION RETRIEVED:")
        print(json.dumps(response.data, indent=2, default=str))
        print("="*80 + "\n")
        return response
    
    def perform_destroy(self, instance):
        # Soft delete
        instance.is_active = False
        instance.save()

class ChatMessageListView(generics.ListAPIView):
    """
    GET: List all messages for a specific chat session
    """
    permission_classes = [IsAuthenticated]
    serializer_class = ChatMessageSerializer
    
    def get_queryset(self):
        session_id = self.kwargs['session_id']
        # Ensure user owns the session
        session = get_object_or_404(ChatSession, id=session_id, user=self.request.user)
        return ChatMessage.objects.filter(session=session)
    
    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        session_id = self.kwargs['session_id']
        print("\n" + "="*80)
        print(f"CHAT MESSAGES RETRIEVED (Session {session_id}):")
        print(json.dumps(response.data, indent=2, default=str))
        print("="*80 + "\n")
        return response


class GenerateChatTitleView(APIView):
    """
    POST: Generate a title for a chat session based on its message history
    
    Request body:
        - session_id: ID of the chat session
    
    Response:
        - session_id: The session ID
        - title: The generated title
        - updated: Whether the session was updated
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        try:
            session_id = request.data.get('session_id')
            
            if not session_id:
                return Response(
                    {"error": "session_id is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get the session (ensure user owns it)
            session = get_object_or_404(ChatSession, id=session_id, user=request.user)
            
            # Get chat history
            messages = ChatMessage.objects.filter(session=session).order_by('created_at')
            
            if not messages.exists():
                return Response(
                    {"error": "No messages found in this session"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Format chat history for the title generation
            chat_history = self._format_chat_history(messages)
            
            # Generate title using OpenAI
            title = self._generate_title(chat_history)
            
            # Update session with new title
            session.title = title
            session.save(update_fields=['title', 'updated_at'])
            
            print(f"Generated title for session {session_id}: {title}")
            
            return Response({
                "session_id": session.id,
                "title": title,
                "updated": True
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            print(f"Error generating chat title: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate title", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _format_chat_history(self, messages, max_messages=6):
        """
        Format chat messages for title generation
        
        Args:
            messages: QuerySet of ChatMessage objects
            max_messages: Maximum number of messages to include
        
        Returns:
            Formatted string of conversation
        """
        formatted = []
        for msg in messages[:max_messages]:
            role = "User" if msg.message_type == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _generate_title(self, chat_history: str) -> str:
        """
        Generate a concise chat title using OpenAI API
        
        Args:
            chat_history: Formatted string of the conversation
        
        Returns:
            A short, descriptive title for the chat session
        """
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a very short, concise and very descriptive title for this conversation 3-6 words. Return only the title, no quotes or punctuation at the end."
                    },
                    {
                        "role": "user",
                        "content": f"Conversation:\n{chat_history}"
                    }
                ],
                temperature=0.7,
                max_tokens=20
            )
            
            title = response.choices[0].message.content.strip()
            # Clean up the title - remove quotes if present
            title = title.strip('"\'')
            return title[:100]  # Limit length just in case
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback: use first user message as title
            return "Untitled Chat"
