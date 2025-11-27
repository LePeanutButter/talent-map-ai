from django.urls import path
from . import views

urlpatterns = [
    path("status/", views.ml_status_view, name="ml_status"),
]