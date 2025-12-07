import threading
from .distilbert_model import JobMatchingService
from .data_loader import load_training_data
import ml.globals as g

def load_and_train():
    try:
        g.ml_status = "training"
        service = JobMatchingService(
            model_name="distilbert-base-uncased",
            save_dir="models/job_matching"
        )

        print("[LM] Loading dataset...")
        train_data = load_training_data("data/training/master_resumes_train.jsonl")
        val_data = load_training_data("data/training/master_resumes_val.jsonl")

        print("[LM] Attempting load or train...")
        service.load_or_train(
            model_id="job_matching_v1",
            train_data=train_data,
            val_data=val_data
        )

        g.job_matching_model = service
        g.ml_status = "ready"
        print("[LM] Model ready.")

    except Exception as e:
        print(f"[LM] Error during model training: {e}")

def start():
    threading.Thread(target=load_and_train, daemon=True).start()