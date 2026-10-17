import os
import math
from flask import Flask, request, jsonify
import joblib

# Initialise Flask
app = Flask(__name__)

# Load model, change accordingly
model_file = os.getenv("MODEL_FILE", "rf_balanced.joblib")

# Safety limit for batch prediction
max_records = int(os.getenv("MAX_RECORDS", "1000"))

# Extra safety for extremely large numeric values
max_abs_value = float(os.getenv("MAX_ABS_VALUE", "1e12"))

def load_model(path: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise SystemExit(
            f"Model file not found: '{path}'. "
            f"Please run the export cell in jobchange.ipynb."
        )
    except Exception as exc:
        raise SystemExit(f"Failed to load model '{path}': {exc}")

model = load_model(model_file)

# Fallback if data does not include feature names
fallback_expected_features = [
    "city_development_index",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "experience",
    "company_size",
    "last_new_job",
    "training_hours",
    "experience_missing",
    "last_new_job_never",
    "last_new_job_unknown",
    "company_type_Early Stage Startup",
    "company_type_Funded Startup",
    "company_type_NGO",
    "company_type_Other",
    "company_type_Public Sector",
    "company_type_Pvt Ltd",
    "company_type_Unknown",
    "major_discipline_Arts",
    "major_discipline_Business Degree",
    "major_discipline_Humanities",
    "major_discipline_No Major",
    "major_discipline_Other",
    "major_discipline_STEM",
    "major_discipline_Unknown",
]


def get_expected_features(loaded_model):
    feature_names = getattr(loaded_model, "feature_names_in_", None)

    if feature_names is not None and len(feature_names) > 0:
        return [str(name) for name in feature_names]

    return fallback_expected_features


expected_features = get_expected_features(model)
expected_feature_set = set(expected_features)

# If the model knows how many features it expects, verify the feature list length.
model_n_features = getattr(model, "n_features_in_", None)

if model_n_features is not None and len(expected_features) != model_n_features:
    raise SystemExit(
        "Feature list length does not match model.n_features_in_. "
        f"Expected by list: {len(expected_features)}, expected by model: {model_n_features}. "
        "Please update expected_features."
    )


feature_to_index = {feature: idx for idx, feature in enumerate(expected_features)}

# Validation rules
binary_features = {
    "relevent_experience",
    "experience_missing",
    "last_new_job_never",
    "last_new_job_unknown",
}

# All one-hot encoded company_type and major_discipline columns are binary.
for feature in expected_features:
    if feature.startswith("company_type_") or feature.startswith("major_discipline_"):
        binary_features.add(feature)


# One-hot groups should have exactly one active column.
one_hot_groups = {
    "company_type": [
        feature for feature in expected_features
        if feature.startswith("company_type_")
    ],
    "major_discipline": [
        feature for feature in expected_features
        if feature.startswith("major_discipline_")
    ],
}

# Basic numeric sanity checks.
# For enrolled_university, education_level, and company_size,
# exact ordinal mapping is assumed to be done before calling this API.
feature_rules = {
    "city_development_index": {
        "min": 0.0,
        "max": 1.0,
    },
    "experience": {
        "min": 0.0,
        "max": 120.0,
    },
    "training_hours": {
        "min": 0.0,
        "max": 100000.0,
    },
    # Unknown = -1, never = 0, 1, 2, 3, 4, >4 = 5
    "last_new_job": {
        "allowed": {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    },
}


def validate_number(feature: str, value):
    """
    Validate one feature value.

    Rules:
    - Binary features must be 0, 1, true, or false.
    - Other features must be finite int or float.
    - Some features have additional min/max/allowed-value rules.
    """

    # Binary / one-hot features
    if feature in binary_features:
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float)) and math.isfinite(value):
            number = float(value)
            if number in (0.0, 1.0):
                return int(number)

        raise ValueError(f"Feature '{feature}' must be binary: 0 or 1.")

    # Non-binary features should not be boolean
    if isinstance(value, bool):
        raise ValueError(f"Feature '{feature}' must be a number, not boolean.")

    if not isinstance(value, (int, float)):
        raise ValueError(f"Feature '{feature}' must be a number.")

    if not math.isfinite(value):
        raise ValueError(f"Feature '{feature}' must be finite.")

    number = float(value)

    if abs(number) > max_abs_value:
        raise ValueError(f"Feature '{feature}' value is too large.")

    rules = feature_rules.get(feature)

    if rules:
        allowed_values = rules.get("allowed")
        if allowed_values is not None and number not in allowed_values:
            allowed_text = ", ".join(
                str(int(v) if float(v).is_integer() else v)
                for v in sorted(allowed_values)
            )
            raise ValueError(
                f"Feature '{feature}' must be one of: {allowed_text}."
            )

        min_value = rules.get("min")
        max_value = rules.get("max")

        if min_value is not None and number < min_value:
            raise ValueError(f"Feature '{feature}' must be >= {min_value}.")

        if max_value is not None and number > max_value:
            raise ValueError(f"Feature '{feature}' must be <= {max_value}.")

    return number


def validate_one_hot_groups(row: list, record_index: int):
    """
    Ensure each one-hot group has exactly one active category.

    Example:
    - Exactly one company_type_* must be 1.
    - Exactly one major_discipline_* must be 1.
    """

    for group_name, group_features in one_hot_groups.items():
        if not group_features:
            continue

        total = sum(row[feature_to_index[feature]] for feature in group_features)

        if total != 1:
            raise ValueError(
                f"Record {record_index}: exactly one '{group_name}' dummy feature must be 1. "
                f"Current sum is {total}."
            )


def validate_record(record, record_index: int):
    """
    Validate one incoming record.

    Accepted formats:
    1. Dictionary:
       {
           "city_development_index": 0.624,
           "relevent_experience": 1,
           ...
       }

    2. List of numbers in EXPECTED_FEATURES order:
       [0.624, 1, ..., 0]
    """

    if isinstance(record, dict):
        missing_features = [
            feature for feature in expected_features
            if feature not in record
        ]

        unknown_features = [
            key for key in record.keys()
            if key not in expected_feature_set
        ]

        if missing_features:
            raise ValueError(
                f"Record {record_index} is missing required features: {missing_features}"
            )

        if unknown_features:
            raise ValueError(
                f"Record {record_index} contains unknown features: {unknown_features}"
            )

        row = [
            validate_number(feature, record[feature])
            for feature in expected_features
        ]

        validate_one_hot_groups(row, record_index)

        return row

    if isinstance(record, list):
        if len(record) != len(expected_features):
            raise ValueError(
                f"Record {record_index} has {len(record)} values, "
                f"but expected {len(expected_features)} values "
                f"in this exact order: {expected_features}"
            )

        row = [
            validate_number(expected_features[i], value)
            for i, value in enumerate(record)
        ]

        validate_one_hot_groups(row, record_index)

        return row

    raise ValueError(
        f"Record {record_index} must be a JSON object or a list of numbers."
    )


def normalize_prediction_value(value):
    """
    Convert float predictions like 0.0 or 1.0 into int for cleaner JSON output.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    if not isinstance(payload, dict):
        return jsonify({
            "error": "JSON body must be an object containing a 'data' key."
        }), 400

    if "data" not in payload:
        return jsonify({
            "error": "Missing required key: 'data'."
        }), 400

    data = payload["data"]

    # Accept:
    # { "data": { ... one record ... } }
    if isinstance(data, dict):
        records = [data]

    # Accept:
    # { "data": [ { ... }, { ... } ] }
    # or:
    # { "data": [0.624, 1, ..., 0] }
    elif isinstance(data, list):
        if len(data) == 0:
            return jsonify({
                "error": "'data' must not be empty."
            }), 400

        # If the list length equals feature count and all items are primitive values,
        # treat it as one flat record.
        is_flat_single_record = (
            len(data) == len(expected_features)
            and all(not isinstance(item, (dict, list)) for item in data)
        )

        if is_flat_single_record:
            records = [data]
        else:
            records = data

    else:
        return jsonify({
            "error": "'data' must be a JSON object, a list of objects, or a list of numbers."
        }), 400

    if len(records) > max_records:
        return jsonify({
            "error": f"Too many records. Maximum allowed: {max_records}."
        }), 400

    # Validate all records before calling the model.
    try:
        rows = [
            validate_record(record, index)
            for index, record in enumerate(records)
        ]
    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    # Call model only after validation succeeds.
    try:
        predictions = model.predict(rows)
    except Exception as exc:
        return jsonify({
            "error": f"Model prediction failed: {exc}"
        }), 500

    predictions = [
        normalize_prediction_value(value)
        for value in predictions.tolist()
    ]

    return jsonify({
        "prediction": predictions,
        "count": len(predictions)
    })


# Health endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_file": model_file,
        "n_features": len(expected_features)
    })


# Main app
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}

    app.run(host="0.0.0.0", port=port, debug=debug)