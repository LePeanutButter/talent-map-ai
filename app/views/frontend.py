from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

def home(request: HttpRequest) -> HttpResponse:
    """
    Renders the home page.

    This view handles HTTP requests to the root or home URL of the application
    and returns the rendered 'home.html' template.

    Args:
        request (HttpRequest): The incoming HTTP request object.

    Returns:
        HttpResponse: A response object containing the rendered HTML content
                      of the home page.
    """
    return render(request, 'home.html')
