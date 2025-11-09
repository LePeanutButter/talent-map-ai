from django.urls import path
from ..views import frontend

"""
URL configuration for the frontend application.

This module maps URL paths to their corresponding view functions.
It defines the routes that handle HTTP requests for the frontend.

Attributes:
    urlpatterns (list): A list of URL pattern objects that Django uses to
                        route incoming requests to the appropriate view.
"""

urlpatterns = [
    path("", frontend.home, name="home"),
]
