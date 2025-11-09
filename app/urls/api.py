from django.urls import path
from ..views import api

"""
URL configuration for the API application.

This module maps URL paths to their corresponding API view functions.
It defines the routes that handle HTTP requests for API endpoints.

Attributes:
    urlpatterns (list): A list of URL pattern objects that Django uses to
                        route incoming API requests to the appropriate view.
"""

urlpatterns = [
    path("resume/", api.extract_view, name="extract"),
]
