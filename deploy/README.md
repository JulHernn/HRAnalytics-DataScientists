# Deployment API

This deployment serves the trained model through a simple Flask API.

> Note: The API assumes the data payload is preprocessed beforehand.

The `testing_deploy.py` file loads the saved model when the application starts. It exposes a `/predict` endpoint that receives preprocessed input data in JSON format. Before making a prediction, the API checks whether the request is valid, including required fields, correct data types, and expected feature structure. If the input is invalid, the API returns an error without calling the model. If the input is valid, the API sends the data to the model and returns the prediction result.

A `/health` endpoint is also available to check whether the service is running.

## Run

```bash
python testing_deploy.py
```

## Endpoints

- `POST /predict` — returns model predictions.
- `GET /health` — checks service status.

## Expected Request Format

```json
{
    "data": [
        {
            "feature_1": 0,
            "feature_2": 1
        }
    ]
}
```

## Expected Response Format

```json
{
    "prediction": [1],
    "count": 1
}
```

## Usage Example

### Payload

File: data.json

```json
{
    "data": [
        {
            "city_development_index": 0.624,
            "relevent_experience": 1,
            "enrolled_university": 0,
            "education_level": 3,
            "experience": 4.0,
            "company_size": 4,
            "last_new_job": 1,
            "training_hours": 30,
            "experience_missing": 0,
            "last_new_job_never": 0,
            "last_new_job_unknown": 0,
            "company_type_Early Stage Startup": 0,
            "company_type_Funded Startup": 0,
            "company_type_NGO": 0,
            "company_type_Other": 0,
            "company_type_Public Sector": 0,
            "company_type_Pvt Ltd": 1,
            "company_type_Unknown": 0,
            "major_discipline_Arts": 0,
            "major_discipline_Business Degree": 0,
            "major_discipline_Humanities": 0,
            "major_discipline_No Major": 0,
            "major_discipline_Other": 0,
            "major_discipline_STEM": 1,
            "major_discipline_Unknown": 0
        }
    ]
}
```

### Command

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @data.json
```

### Response

```json
{
    "count":1,
    "prediction":[1]
}
```