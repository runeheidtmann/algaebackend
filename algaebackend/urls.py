from django.contrib import admin
from django.urls import path
from algaebackend import views
from django.conf import settings
from django.conf.urls.static import static

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from .views import DocumentUploadView
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/ragchat/',views.RagChatAPIView.as_view(),name='ragchat-api'),
    path('api/chat/',views.ChatAPIView.as_view(),name='chat-api'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/userdata/', views.CurrentUserView.as_view(), name='current_user'),
    path('api/register/', views.UserRegistrationView.as_view(), name='user_registration'),
    path('api/evaluations/', views.EvaluationView.as_view(), name='evaluation-list'),
    path('api/evaluations/<int:id>', views.EvaluationDetailView.as_view(), name='evaluation-detail'),
    path('api/upload-document/', DocumentUploadView.as_view(), name='upload-document'),
    path('api/documents/', views.DocumentsView.as_view(), name='documents'),
    
    # Chat Session Management
    path('api/chat-sessions/', views.ChatSessionListCreateView.as_view(), name='chat-sessions-list'),
    path('api/chat-sessions/<int:pk>/', views.ChatSessionDetailView.as_view(), name='chat-session-detail'),
    path('api/chat-sessions/<int:session_id>/messages/', views.ChatMessageListView.as_view(), name='chat-messages'),
    path('api/chat-sessions/generate-title/', views.GenerateChatTitleView.as_view(), name='generate-chat-title'),
    
    # Alternative URL pattern for frontend compatibility
    path('api/chat/session/<int:pk>/', views.ChatSessionDetailView.as_view(), name='chat-session-detail-alt'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)