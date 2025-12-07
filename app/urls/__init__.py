from django.urls import path, include

"""
Root URL configuration for the project.

This module aggregates URL patterns from different sub-applications
(frontend and API) into a single entry point for Django's URL routing system.

Attributes:
    urlpatterns (list): A list of URL pattern objects that include
                        sub-application URL configurations.
                        - The root path ("") includes the frontend app URLs.
                        - The "api/" path includes the API app URLs.
"""

urlpatterns = [
    path("", include("app.urls.frontend")),
    path("api/", include("app.urls.api")),
]
