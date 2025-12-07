from django.http import JsonResponse
import ml.globals as g

def ml_status_view(request):
    """
    Returns whether the ML model is loaded and ready.
    """
    status = getattr(g, "ml_status", "training")
    return JsonResponse({"status": status})
