import os
from . import ml_thread
from django.apps import AppConfig

class MLConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ml"

    def ready(self):
        """
        This runs once when Django starts.
        Only run in the *actual server process*, not the autoreloader
        """
        if os.environ.get("RUN_MAIN") != "true":
            return
        ml_thread.start()
