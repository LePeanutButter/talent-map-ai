# TalentMap AI: Ethical Talent Matching with NLP and BERT

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Ethical and technical AI application for matching job candidates with vacancies through semantic analysis.
TalentMap AI leverages Natural Language Processing (NLP) and Transformer-based embeddings (DistilBERT) to evaluate compatibility between résumés and job descriptions beyond keyword matching.
The project emphasizes fairness, transparency, and interpretability, addressing algorithmic bias while improving employment outcomes.

Developed for the *Principles of Artificial Intelligence Technologies (PTIA)* course at **Escuela Colombiana de Ingeniería Julio Garavito**.

---

## Table of Contents

* [Background](#background)
* [Install](#install)
* [Usage](#usage)
* [Architecture](#architecture)
* [Ethical Framework](#ethical-framework)
* [Test Results](#test-results)
* [Maintainers](#maintainers)
* [License](#license)

---

## Background

Modern recruitment systems often rely on keyword-based matching, which fails to capture the deeper semantic relationships between a candidate’s skills and a job’s requirements.
**TalentMap AI** addresses this limitation by combining **machine learning**, **semantic embeddings**, and **ethical AI design** to improve job–candidate compatibility.

This project aims to:

1. Develop a semantic model using **DistilBERT** for candidate–job matching.
2. Integrate **fairness and bias auditing** tools to ensure responsible AI behavior.
3. Provide a **web-based MVP** demonstrating real-time compatibility scoring.

---

## Install

This project requires **Python 3.10+** and the following key dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies

* `Django` – Web backend and API.
* `transformers` – DistilBERT-based language models.
* `jQuery` – For front-end interactivity.

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/LePeanutButter/talent-map-ai.git
   cd talent-map-ai
   ```

2. Run the Django server:

   ```bash
   python manage.py runserver
   ```

3. Upload a résumé through the web interface.
   The model will analyze and return a **semantic compatibility score** between the uploaded résumé and job descriptions.

---

## Architecture

The TalentMap AI system follows a modular architecture:

* **Frontend (SPA)** – HTML, CSS, JS for visualization of recommendations, using jQuery for interactivity.
* **Backend (Django REST)** – API for résumé and job description processing.
* **AI Engine** – DistilBERT embeddings for semantic similarity.
* **Ethics Layer** – Bias detection, anonymization, and explainability mechanisms.

---

## Ethical Framework

TalentMap AI follows UNESCO’s *Recommendation on the Ethics of Artificial Intelligence (2021)*, applying principles of:

* **Fairness** – Avoiding bias by anonymizing and auditing datasets.
* **Transparency** – Explaining how recommendations are generated.
* **Accountability** – Ensuring human oversight and responsible AI design.

---

## Test Results

These results are based on a series of **test runs** performed to demonstrate how the model works when trained with a small dataset and tested on a larger, more representative dataset.

### Training Configuration:

The model was trained with the following configuration:

```
Training Configuration:
Model ID: test_model
Mode: cosine
Epochs: 2
Batch size: 4
Learning rate: 0.0002
Freeze BERT: True
Device: cpu
Training samples: 8
Validation samples: 2
```

### Training Progress:

* **Epoch 1/2**:

  * Train Loss: 0.3157
  * Val Loss: 0.3781
* **Epoch 2/2**:

  * Train Loss: 0.2045
  * Val Loss: 0.3811

The model training took **173.68 seconds**, and the model was successfully saved:

* **Compressed model saved to**: `test_model/test_model_cosine_20251125_173520.pt.xz`
* **Model saved to**: `test_model/test_model_cosine_20251125_173520.pt.xz`

### Model Testing:

1. **Model Loading and Inference:**

   After training, the model was reloaded for testing and evaluated for similarity:

   * Original model similarity: `0.9317`
   * Loaded model similarity: `0.9317`
   * **Difference**: `0.000000` (the models are identical after reloading)

2. **Batch Prediction Testing:**

   Batch predictions were completed in **0.0521 seconds** with the following results:

   | Score  | Job Title               | Resume Description                    |
   | ------ | ----------------------- | ------------------------------------- |
   | 0.8458 | Python ML engineer...   | Expert in Python and machine learning |
   | 0.6147 | Marketing manager...    | Software engineer with 5 years...     |
   | 0.6766 | Data scientist...       | PhD in statistics, ML experience...   |
   | 0.6620 | Sales representative... | Frontend developer...                 |

3. **Model Parameter Validation:**

   * SUCCESS: Loaded model parameters match the saved model.

4. **Test Completion:**

   * All tests completed successfully in **190.272s**.

---

### Testing on HF Resumes Dataset:

For a broader evaluation, the model was tested using a dataset of **4817 resumes**. Here are the **Top 10 most relevant matches** according to the model:

1. **Row 2419** | Score: 0.9268 — Creative iOS Developer with a focus on building elegant and user-friendly applications for Apple devices. Proficient in Swift, Xcode, and iOS SDK.
2. **Row 1569** | Score: 0.9234 — Creative Web Developer with a passion for building responsive and user-friendly web applications. Proficient in HTML, CSS, JavaScript, and front-end frameworks like React.
3. **Row 1901** | Score: 0.9230 — AI Engineer with expertise in artificial intelligence, machine learning, and deep learning. Skilled in Python, TensorFlow, and developing AI-powered applications.
4. **Row 1670** | Score: 0.9228 — Analytical Data Scientist with a passion for machine learning, statistical analysis, and data-driven decision making. Experienced with Python, R, and modern data visualization tools.
5. **Row 5**   | Score: 0.9224 — Python Developer in the field of computer vision for a US-based client in the banking domain. Design and development of computer vision-based algorithms.
6. **Row 108** | Score: 0.9224 — Python Developer in the field of computer vision for a US-based client in the banking domain. Design and development of computer vision-based algorithms.
7. **Row 2614**| Score: 0.9220 — Innovative Android Developer with a passion for mobile app development and user experience design. Skilled in Java, Kotlin, and Android Studio.
8. **Row 2480**| Score: 0.9213 — Creative iOS Developer with a focus on building elegant and user-friendly applications for Apple devices. Proficient in Swift, Xcode, and iOS SDK.
9. **Row 71**  | Score: 0.9208 — Python Developer in the field of computer vision for a US-based client in the banking domain. Design and development of computer vision-based algorithms.
10. **Row 175**| Score: 0.9208 — Python Developer in the field of computer vision for a US-based client in the banking domain. Design and development of computer vision-based algorithms.

### Important Notes on Test Results:

The test results above show strong matching scores across the top resumes, but it's important to highlight the following:

1. **Dataset Size and Quality:** While testing with a larger dataset of 4817 resumes provides more context, the model was trained on a very small dataset (only 8 training samples). This can still impact the accuracy and relevance of the results, especially in larger or more complex real-world scenarios.

2. **No Anonymization in Testing:** The model has not yet fully implemented anonymization, which could influence its ability to avoid potential biases in resume matching. Further improvements on anonymization and fairness are planned for future updates.

---

## Maintainers

* [andrescalderonr](https://github.com/andrescalderonr) - Andrés Felipe Calderón Ramírez
* [LePeanutButter](https://github.com/LePeanutButter) - Santiago Botero Garcia

---

## License

[MIT](LICENSE) © 2025 TalentMap AI Team

This README follows the <u>**Standard Readme**</u> specification.
