# TalentMap AI: Ethical Talent Matching with DistilBERT

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Ethical and technical AI application for matching job candidates with vacancies through semantic analysis.
TalentMap AI leverages Transformer-based embeddings (DistilBERT) to evaluate compatibility between résumés and job descriptions beyond keyword matching.
The project emphasizes fairness, transparency, and interpretability, addressing algorithmic bias while improving employment outcomes.

Developed for the *Principles of Artificial Intelligence Technologies (PTIA)* course at **Escuela Colombiana de Ingeniería Julio Garavito**.

## Table of Contents

* [Background](#background)
* [Install](#install)
   * [Main Dependencies](#main-dependencies)
* [Usage](#usage)
* [Architecture](#architecture)
* [Ethical Framework](#ethical-framework)
* [Test Results](#test-results)
   * [Training Configuration](#training-configuration)
   * [Training Progress](#training-progress)
   * [Model Testing](#model-testing)
* [Maintainers](#maintainers)
* [Contributors](#contributors)
* [License](#license)

## Background

Modern recruitment systems often rely on keyword-based matching, which fails to capture the deeper semantic relationships between a candidate’s skills and a job’s requirements.
**TalentMap AI** addresses this limitation by combining **machine learning**, **semantic embeddings**, and **ethical AI design** to improve job-candidate compatibility.

This project aims to:

1. Develop a semantic model using **DistilBERT** for candidate-job matching.
2. Integrate **fairness and bias auditing** tools to ensure responsible AI behavior.
3. Provide a **web-based MVP** demonstrating real-time compatibility scoring.

## Install

This project requires **Python 3.10+** and the following key dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies

* `Django` - Web backend and API.
* `transformers` - DistilBERT-based language models.
* `jQuery` - For front-end interactivity.

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

## Architecture

The TalentMap AI system follows a modular architecture:

* **Frontend (SPA)** - HTML, CSS, JS for visualization of recommendations, using jQuery for interactivity.
* **Backend (Django REST)** - API for résumé and job description processing.
* **AI Engine** - DistilBERT embeddings for semantic similarity.
* **Ethics Layer** - Bias detection, anonymization, and explainability mechanisms.

## Ethical Framework

TalentMap AI follows UNESCO’s *Recommendation on the Ethics of Artificial Intelligence (2021)*, applying principles of:

* **Fairness** - Avoiding bias by anonymizing and auditing datasets.
* **Transparency** - Explaining how recommendations are generated.
* **Accountability** - Ensuring human oversight and responsible AI design.

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

## Maintainers

* [LePeanutButter](https://github.com/LePeanutButter) - Santiago Botero Garcia

## Contributors

This project exists thanks to all the people who contribute.

* [andrescalderonr](https://github.com/andrescalderonr) - Andrés Felipe Calderón Ramírez
* [LePeanutButter](https://github.com/LePeanutButter) - Santiago Botero Garcia

## License

[MIT](LICENSE) © 2025 TalentMap AI Team

---

This README follows the <u>**Standard Readme**</u> specification.
