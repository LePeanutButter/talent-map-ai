# TalentMap AI: Ethical Talent Matching with NLP and BERT

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Ethical and technical AI application for matching job candidates with vacancies through semantic analysis.  
TalentMap AI leverages Natural Language Processing (NLP) and Transformer-based embeddings (BERT, Word2Vec, spaCy) to evaluate compatibility between résumés and job descriptions beyond keyword matching.  
The project emphasizes fairness, transparency, and interpretability, addressing algorithmic bias while improving employment outcomes.

Developed for the *Principles of Artificial Intelligence Technologies (PTIA)* course at **Escuela Colombiana de Ingeniería Julio Garavito**.

---

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Architecture](#architecture)
- [Ethical Framework](#ethical-framework)
- [Maintainers](#maintainers)
- [License](#license)

---

## Background

Modern recruitment systems often rely on keyword-based matching, which fails to capture the deeper semantic relationships between a candidate’s skills and a job’s requirements.  
**TalentMap AI** addresses this limitation by combining **machine learning**, **semantic embeddings**, and **ethical AI design** to improve job–candidate compatibility.

This project aims to:
1. Develop a semantic model using **BERT** and **Word2Vec** for candidate–job matching.  
2. Integrate **fairness and bias auditing** tools to ensure responsible AI behavior.  
3. Provide a **web-based MVP** demonstrating real-time compatibility scoring.  

---

## Install

This project requires **Python 3.10+** and the following key dependencies:

```bash
pip install -r requirements.txt
````

### Main Dependencies

* `Django` – Web backend and API.
* `spaCy` – NLP preprocessing.
* `scikit-learn` – Classical ML models and evaluation.
* `Keras / TensorFlow` – Deep learning and embeddings.
* `transformers` – BERT-based language models.

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

* **Frontend (SPA)** – HTML, CSS, JS for visualization of recommendations.
* **Backend (Django REST)** – API for résumé and job description processing.
* **AI Engine** – BERT/Word2Vec embeddings for semantic similarity.
* **Ethics Layer** – Bias detection, anonymization, and explainability mechanisms.

---

## Ethical Framework

TalentMap AI follows UNESCO’s *Recommendation on the Ethics of Artificial Intelligence (2021)*, applying principles of:

* **Fairness** – Avoiding bias by anonymizing and auditing datasets.
* **Transparency** – Explaining how recommendations are generated.
* **Accountability** – Ensuring human oversight and responsible AI design.

---

## Maintainers

* [andrescalderonr](https://github.com/andrescalderonr) - Andrés Felipe Calderón Ramírez
* [LePeanutButter](https://github.com/LePeanutButter) - Santiago Botero Garcia

---

## License

[MIT](LICENSE) © 2025 TalentMap AI Team

This README follows the <u>**Standard Readme**</u> specification.
