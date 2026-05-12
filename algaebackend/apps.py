from django.apps import AppConfig


class AlgaebackendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'algaebackend'

    def ready(self):
        # Import signals to register them
        import algaebackend.signals  # noqa
