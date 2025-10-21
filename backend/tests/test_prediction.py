import pytest


def test_prediction_validation_error(client):
    # Missing required fields; expect 422
    resp = client.post('/api/v1/prediction', json={"age": 50})
    assert resp.status_code in {400, 422}


def test_prediction_happy_path(client):
    payload = {
        "age": 55,
        "is_male": True,
        "chest_pain_type": "typical_angina",
        "resting_blood_pressure": 130,
        "serum_cholesterol": 245,
        "fasting_blood_sugar_over_120": False,
        "resting_ecg_results": "normal",
        "maximum_heart_rate_achieved": 150,
        "exercise_induced_angina": False,
        "st_depression_exercise": 1.0,
        "st_slope_peak_exercise": "upsloping",
        "major_vessels_colored": 0,
        "thalassemia_type": "normal"
    }
    resp = client.post('/api/v1/prediction', json=payload)
    assert resp.status_code in {200, 201}
    body = resp.json()
    # Response schema keys may vary; check common signals
    assert 'risk_probability' in body or 'probability' in body or 'confidence_score' in body
