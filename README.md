# Multimodal Pneumonia Prediction

![CI](https://github.com/doug0x2a/mit-cv-project/actions/workflows/ci.yml/badge.svg)

This repository contains a machine learning project focused on predicting pneumonia using chest X-ray images and associated tabular clinical data.

At this stage, the repository includes:
- A deployed API service built with FastAPI for serving multimodal predictions.
- Unit tests and a CI pipeline for validating API functionality.

I plan to add the initial model training and data processing code in future updates.

---

## Quick Start

### Run the API locally
```bash
uvicorn app.main:app --reload
```

### Run tests
```bash
pytest tests/
```

