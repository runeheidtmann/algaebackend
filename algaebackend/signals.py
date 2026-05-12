from django.core.mail import send_mail
from django.dispatch import receiver
from django.template.loader import render_to_string
from django.conf import settings
from django_rest_passwordreset.signals import reset_password_token_created


@receiver(reset_password_token_created)
def password_reset_token_created(sender, instance, reset_password_token, *args, **kwargs):
    """
    Handles password reset tokens.
    When a token is created, an email is sent to the user.
    """
    # Get the frontend URL from settings
    frontend_url = getattr(settings, 'FRONTEND_URL', 'https://algaebrain.dk')
    reset_url = f"{frontend_url}/reset-password?token={reset_password_token.key}"
    
    # Email subject
    subject = "Password Reset for AlgaeBrain"
    
    # Plain text email body
    message = f"""
Hello {reset_password_token.user.username},

You have requested to reset your password for AlgaeBrain.

Click the link below to reset your password:
{reset_url}

If you did not request this password reset, please ignore this email.

This link will expire in 24 hours.

Best regards,
AlgaeBrain Team
"""
    
    # Send the email
    send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[reset_password_token.user.email],
        fail_silently=False,
    )
