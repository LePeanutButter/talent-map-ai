from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from ..services.document_processing_service import DocumentProcessingService
import ml.globals as g

@csrf_exempt
def extract_view(request):
    """
    Endpoint to extract text from multiple files sent via POST and predict similarity with job description.

    Workflow:
        1. Checks that the request method is POST.
        2. Reads the uploaded files content and determines their extensions.
        3. Prepares metadata including a demo document ID, user ID, original filenames, and file sizes.
        4. Uses TextExtractor to extract text from the files' content.
        5. Uses the ML model to predict similarity with a job description (provided in the request body).
        6. Returns a JSON response with the extracted text from each file, prediction score, or an error message.

    Args:
        request (HttpRequest): The incoming HTTP request object.

    Returns:
        JsonResponse: A JSON object containing either the extracted text, prediction scores, or an error message.
            The HTTP status code is determined as follows:
            - 200: Extraction and prediction succeeded
            - 400: Bad request (missing file, exceeding file count, or reading error)
            - 405: Method not allowed (non-POST request)
            - 500: Internal extraction error or ML error
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST method only allowed"}, status=405)

    job_text = request.POST.get("job_text")
    if not job_text:
        return JsonResponse({"error": "Job description (job_text) is required in the request body."}, status=400)

    uploaded_files = request.FILES.getlist("file")
    if not uploaded_files:
        return JsonResponse({"error": "No file received."}, status=400)
    if len(uploaded_files) > 10:
        return JsonResponse({"error": "You can upload a maximum of 10 files."}, status=400)

    service = DocumentProcessingService()
    extracted_texts = []
    prediction_scores = []
    status_code = 200

    for uploaded_file in uploaded_files:
        result = service.process_uploaded_file(
            uploaded_file=uploaded_file,
            user_id="guest",
            doc_id=None
        )

        status_code = result.pop("status_code", status_code)
        extracted_text = result.get("data", {}).get("extracted_text", "")

        if not extracted_text:
            return JsonResponse({"error": "Failed to extract text from one of the files."}, status=500)

        try:
            prediction_score = g.job_matching_model.predict(job_text, extracted_text, mode="cosine")
        except Exception as e:
            return JsonResponse({"error": f"Error during prediction: {str(e)}"}, status=500)

        extracted_texts.append(extracted_text)
        prediction_scores.append(prediction_score)

    return JsonResponse({
        "extracted_texts": extracted_texts,
        "prediction_scores": prediction_scores
    }, status=status_code)