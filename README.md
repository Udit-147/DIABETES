# GlucoCare — Diabetes Risk Screening Demo

Responsive HTML/CSS/JavaScript frontend connected to a Flask prediction API.

## Files
- `app.py` — Flask routes and model inference
- `mymodel.pkl` — supplied trained model
- `templates/index.html` — page markup
- `static/style.css` — responsive styling
- `static/app.js` — form behavior and API integration

## Run locally (Windows)
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

## Render deployment
Build command: `pip install -r requirements.txt`
Start command: `gunicorn app:app`

Ensure `mymodel.pkl` is at the project root. If the model depends on extra libraries (for example XGBoost), add the matching package and ideally the training-time version to `requirements.txt`.

## API
- `GET /api/health`
- `POST /api/predict` JSON fields: `Glucose`, `BloodPressure`, `Insulin`, `BMI`, `Age`

Feature order passed to model: Glucose, BloodPressure, Insulin, BMI, Age. If training used scaling/imputation/pipeline preprocessing, replicate that preprocessing before inference.

## Important
Only load pickle files from trusted sources; deserialization can execute code. This is a demo, not a medical device or diagnosis. Predictions can be wrong and must not guide treatment decisions.
