import json
import unittest
from django.test import TestCase
from app.services.job_matching_service import JobMatchingService
from pathlib import Path

class JobMatchingDatasetTest(TestCase):
    """
    Test suite for validating job–resume matching using a real dataset.

    This test case loads a subset of resumes from the master_resumes.jsonl dataset
    and evaluates the output produced by the JobMatchingService. It verifies
    structural correctness, score validity, and provides a preview of the
    highest-ranking match results.
    """

    def setUp(self):
        """
        Prepare the test environment by loading sample resumes
        and defining a representative job offer.

        Workflow:
            1. Initialize JobMatchingService
            2. Load a controlled number of resumes from disk
            3. Convert each JSONL line into a Python dict
            4. Prepare a standardized job description for testing
        """
        self.service = JobMatchingService()

        dataset_path = Path("data/master_resumes.jsonl")

        self.cv_samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 4817:
                    break
                try:
                    self.cv_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"JSON decoding error at line {i}")

        self.job_offer = """
        We are looking for a Python Developer experienced in Django, REST APIs,
        and machine learning with TensorFlow. Knowledge of OpenCV and cloud
        deployment is a plus.
        """

    def test_real_dataset(self):
        """
        Validate job–resume matching against the real master_resumes.jsonl dataset.

        This test ensures:
            - The service returns a list of results
            - At least one resume match is produced
            - All match scores fall within the 0–100% range
            - The top results are printed for debugging and manual inspection

        The test evaluates correctness of structure, not the semantic
        correctness of the ranking model itself.
        """
        results = self.service.compute_match(self.job_offer, self.cv_samples)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        print("\nTop 10 Job Matching Results:")
        for r in results[:10]:
            print(f"CV {r['resume_index']} → {r['match_score']}% match")

        for r in results:
            self.assertGreaterEqual(r["match_score"], 0)
            self.assertLessEqual(r["match_score"], 100)

if __name__ == '__main__':
    unittest.main()
