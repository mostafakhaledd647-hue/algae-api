# Algae Image Classification API

Flask REST API for classifying algae images using a deep learning model (ResNet101V2).

---

## 🚀 API Endpoint

### POST /predictApi

Classifies an uploaded algae image.

---

## 📥 Request Format

- Method: POST
- Body: form-data
- Key: `fileup`
- Value: Image file (jpg / png)

---

## 📤 Response Format

```json
{
  "prediction": "Microcystis",
  "confidence": 0.97
}
