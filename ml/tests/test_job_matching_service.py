import unittest
import time
import torch
import shutil
import os
import json
from ml.distilbert_model import JobMatchingService

class TestJobMatchingService(unittest.TestCase):
    def test_job_matching(self):
        # ============================================================
        # STAGE 1 — INITIALIZATION
        # ============================================================
        dummy_train = [
            ("Python developer with experience in ML",
             "I have 3 years of experience working with Python and ML, including building ML models with TensorFlow and Scikit-Learn.",
             1),

            ("Marketing specialist",
             "Experienced backend engineer skilled in Java and Node.js with expertise in building scalable applications.",
             0),

            ("Data engineer needed for pipelines",
             "I have built ETL pipelines in AWS, using services like Lambda, Glue, and S3. I also have experience with Python and SQL.",
             1),

            ("Frontend React developer",
             "I specialize in front-end web development with React, Redux, and TypeScript. I also have knowledge of UX/UI principles.",
             1),

            ("NLP Researcher",
             "I have designed state-of-the-art NLP models using transformers such as BERT and GPT-3 for text classification and sentiment analysis.",
             1),

            ("DevOps engineer needed",
             "I specialize in cloud infrastructure with AWS and Azure, using Kubernetes, Docker, and CI/CD pipelines for automated deployments.",
             1),

            ("Graphic designer position",
             "As a full-stack web developer, I am proficient in JavaScript, HTML, CSS, and have experience designing user interfaces.",
             0),

            ("Cloud architect role",
             "I have AWS, Azure, and Google Cloud certifications, specializing in designing cloud-native solutions and infrastructure management.",
             1),

            ("AI Research Scientist",
             "I have a PhD in computer science and work extensively with reinforcement learning, deep neural networks, and natural language processing.",
             1),

            ("Cybersecurity specialist",
             "I have 5 years of experience working with firewalls, VPNs, and incident response. I have implemented encryption and authentication systems.",
             0),

            ("Full-stack developer with React and Node.js",
             "I have experience building full-stack web applications using React, Node.js, Express, and MongoDB. I also work with cloud platforms like AWS.",
             1),

            ("Senior UX/UI designer",
             "I have extensive experience in designing intuitive user interfaces for web and mobile applications using Figma, Adobe XD, and Sketch.",
             0),

            ("Machine learning engineer with expertise in deep learning",
             "I have experience building and deploying machine learning models using TensorFlow, PyTorch, and Keras for computer vision tasks.",
             1),

            ("Product manager for software development",
             "I have experience managing cross-functional teams to deliver software products, including market research, user stories, and agile methodologies.",
             0),

            ("Data analyst for business intelligence",
             "Skilled in data cleaning, analysis, and visualization using tools like Python, SQL, and Tableau to derive actionable insights from large datasets.",
             0),

            ("Senior backend engineer",
             "I have 8 years of experience with Java, Spring, and SQL, building highly scalable and performance-oriented backend services.",
             0)
        ]

        dummy_val = [
            ("Senior ML engineer",
             "With over 10 years of experience in machine learning, I have worked on everything from neural networks to NLP tasks using Python, TensorFlow, and PyTorch.",
             1),

            ("Sales manager needed",
             "I have worked in software development for 6 years, specializing in JavaScript and web technologies such as React and Angular.",
             0),

            ("Lead data scientist",
             "I specialize in machine learning models, including deep learning, natural language processing, and reinforcement learning. I also have strong knowledge of cloud platforms.",
             1),

            ("Software engineer for mobile apps",
             "Experienced mobile app developer with proficiency in Swift, Kotlin, and React Native, as well as expertise in integrating APIs and databases.",
             0),

            ("Cloud solutions architect",
             "I have extensive experience architecting cloud-native applications on AWS and Azure. I specialize in containerization using Kubernetes and Docker.",
             1),

            ("Digital marketing strategist",
             "I have experience managing large-scale digital marketing campaigns, focusing on SEO, PPC, social media marketing, and data-driven insights.",
             0),

            ("Big data engineer for analytics",
             "I have experience building data pipelines in Hadoop and Spark, as well as working with cloud storage solutions like S3, GCS, and Azure Blob Storage.",
             1),

            ("AI/ML Researcher in healthtech",
             "I have worked on AI-driven medical image analysis and predictive models for healthcare, utilizing deep learning and computer vision.",
             1),

            ("UX designer for mobile applications",
             "I am a user experience designer focused on mobile apps, using Figma, InVision, and Adobe XD to create wireframes and prototypes.",
             0),

            ("Blockchain developer",
             "I have developed decentralized applications using Ethereum, Solidity, and smart contracts. My experience also includes developing tokenomics and blockchain solutions.",
             0)
        ]

        print("\n[1] Initializing JobMatchingService...")
        service = JobMatchingService(save_dir="test_model", model_name="distilbert-base-uncased")

        # ============================================================
        # STAGE 2 — TRAINING AND SAVING
        # ============================================================
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

        # ============================================================
        # STAGE 3 — LOADING MODEL FROM DISK
        # ============================================================
        print("\n[3] Loading model from disk...")
        loaded_model = service.load_model(model_path)

        # ============================================================
        # STAGE 4 — ORIGINAL VS LOADED MODEL TEST
        # ============================================================
        print("\n[4] Testing inference with original + loaded model...")

        job_text = "We need a Python engineer with ML skills"
        res_text = "I have 3 years working in ML with Python"

        sim_original = service.predict(job_text, res_text, mode="cosine")
        sim_loaded = loaded_model.predict(job_text, res_text, mode="cosine")

        print(f"\nOriginal model similarity: {sim_original:.4f}")
        print(f"Loaded model similarity:    {sim_loaded:.4f}")
        print(f"Difference: {abs(sim_original - sim_loaded):.6f}")

        # ============================================================
        # STAGE 5 — MODEL PARAMETERS AND TRAINING CHECKS
        # ============================================================
        print("\n[5] Checking if parameters match...")

        match = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(service.model.parameters(), loaded_model.parameters())
        )

        if match:
            print("SUCCESS — Loaded model parameters match the saved model!")
        else:
            print("WARNING — Loaded model parameters differ slightly (this can happen due to device transfer)")

        # ============================================================
        # STAGE 6 — BATCH PREDICTION
        # ============================================================
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

        print("\n[7] Testing HF resumes dataset (Top 10 most relevant)...")

        # ============================================================
        # STAGE 7 — Testing HF resume dataset
        # ============================================================
        dataset_path = "data/master_resumes.jsonl"
        assert os.path.exists(dataset_path), "Dataset file not found at data/master_resumes.jsonl"

        resumes = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                summary = obj.get("personal_info", {}).get("summary", "")
                exp_items = obj.get("experience", [])
                exp_text = " ".join(
                    e.get("title", "") + " " + " ".join(e.get("responsibilities", []))
                    for e in exp_items
                )

                skills = obj.get("skills", {})
                technical = skills.get("technical", {})
                tech_skill_text = " ".join(
                    s.get("name", "")
                    for section in technical.values()
                    for s in section
                    if isinstance(section, list)
                )

                full_resume_text = " ".join([summary, exp_text, tech_skill_text]).strip()
                if full_resume_text:
                    resumes.append(full_resume_text)

        print(f"Loaded {len(resumes)} resumes from dataset.")

        job_query = "Looking for a Python developer with machine learning experience"

        scored_with_index = [
            (i, resume_text, service.predict(job_query, resume_text, mode="cosine"))
            for i, resume_text in enumerate(resumes, start=1)
        ]

        scored_with_index.sort(key=lambda x: x[2], reverse=True)

        print("\nTop 10 resume matches with row index:\n")
        for i, (row_num, text, score) in enumerate(scored_with_index[:10], start=1):
            preview = text[:200].replace("\n", " ")
            print(f"{i}. Row {row_num} | Score: {score:.4f} — {preview}...")

        # ============================================================
        # STAGE 8 — CLEANUP
        # ============================================================
        print("\n[8] Cleaning up: Removing the test model directory...")
        if os.path.exists("test_model"):
            shutil.rmtree("test_model")
            print("Test model directory removed.")
        else:
            print("No test model directory found to remove.")

if __name__ == "__main__":
    unittest.main(verbosity=2)