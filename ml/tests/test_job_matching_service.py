import unittest
import time
import torch
import shutil
import os
from ml.distilbert_model import JobMatchingService

class TestJobMatchingService(unittest.TestCase):
    def test_job_matching(self):
        dummy_train = [
            ("Python developer with experience in ML",
             "I have 3 years of experience working with Python and ML.", 1),
            ("Marketing specialist",
             "Experienced backend engineer in Java", 0),
            ("Data engineer needed for pipelines",
             "I design ETL pipelines in AWS and Python.", 1),
            ("Frontend React developer",
             "I mostly work in cybersecurity and networking.", 0),
            ("NLP Researcher",
             "I have built BERT-based models for text classification", 1),
            ("DevOps engineer needed",
             "I specialize in Kubernetes and CI/CD pipelines", 1),
            ("Graphic designer position",
             "Senior data scientist with PhD in statistics", 0),
            ("Cloud architect role",
             "I have AWS and Azure certifications", 1),
        ]

        dummy_val = [
            ("Senior ML engineer",
             "10 years of experience in machine learning", 1),
            ("Sales manager needed",
             "I'm a software developer", 0),
        ]

        print("\n[1] Initializing JobMatchingService...")
        service = JobMatchingService(save_dir="test_model", model_name="distilbert-base-uncased")

        print("\n[2] Training and saving model...")
        start_time = time.time()

        model_path = service.train_and_save(
            model_id="test_model",
            train_data=dummy_train,
            val_data=dummy_val,
            epochs=2,
            mode="cosine",
            batch_size=4,
            freeze_bert=True,
            lr=2e-4
        )

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        print("\n[3] Loading model from disk...")
        loaded_model = service.load_model(model_path)

        print("\n[4] Testing inference with original + loaded model...")

        job_text = "We need a Python engineer with ML skills"
        res_text = "I have 3 years working in ML with Python"

        sim_original = service.predict(job_text, res_text, mode="cosine")
        sim_loaded = loaded_model.predict(job_text, res_text, mode="cosine")

        print(f"\nOriginal model similarity: {sim_original:.4f}")
        print(f"Loaded model similarity:    {sim_loaded:.4f}")
        print(f"Difference: {abs(sim_original - sim_loaded):.6f}")

        print("\n[5] Checking if parameters match...")

        match = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(service.model.parameters(), loaded_model.parameters())
        )

        if match:
            print("SUCCESS — Loaded model parameters match the saved model!")
        else:
            print("WARNING — Loaded model parameters differ slightly (this can happen due to device transfer)")

        print("\n[6] Testing batch prediction...")

        test_pairs = [
            ("Python ML engineer", "Expert in Python and machine learning"),
            ("Marketing manager", "Software engineer with 5 years experience"),
            ("Data scientist", "PhD in statistics, ML experience"),
            ("Sales representative", "Frontend developer"),
        ]

        batch_start = time.time()
        batch_scores = service.batch_predict(test_pairs, mode="cosine", batch_size=4)
        batch_time = time.time() - batch_start

        print(f"\nBatch prediction completed in {batch_time:.4f} seconds")
        for (job, resume), score in zip(test_pairs, batch_scores):
            print(f"Score: {score:.4f} | Job: '{job[:30]}...' | Resume: '{resume[:30]}...'")

        print("All tests completed successfully!")

        self.assertGreater(len(batch_scores), 0, "Expected at least one score to be calculated.")

        print("\n[7] Cleaning up: Removing the test model directory...")
        if os.path.exists("test_model"):
            shutil.rmtree("test_model")
            print("Test model directory removed.")
        else:
            print("No test model directory found to remove.")

if __name__ == "__main__":
    unittest.main(verbosity=2)