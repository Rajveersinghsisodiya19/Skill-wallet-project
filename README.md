# 🛡️ MediGuard — Health Insurance Fraud Detection

A full-stack web application that uses a trained **Random Forest** ML model to predict whether a health insurance claim is fraudulent or legitimate.

---

## 📁 Project Structure

```
mediguard/
├── app.py                    ← Flask backend (API + serves frontend)
├── requirements.txt          ← Python dependencies
├── insurance_claims.csv      ← (Optional) Real dataset — place here
│
├── model/
│   ├── train_model.py        ← Train & save the model
│   ├── model.pkl             ← Trained Random Forest (auto-generated)
│   ├── scaler.pkl            ← StandardScaler (auto-generated)
│   ├── num_cols.pkl          ← Numerical column list (auto-generated)
│   └── feature_columns.pkl   ← Feature alignment (auto-generated)
│
└── templates/
    └── index.html            ← Full frontend UI
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Add real dataset
Place `insurance_claims.csv` in the project root folder.  
If not provided, the app trains on synthetic data automatically.

### Step 3 — Train the model
```bash
python model/train_model.py
```
This generates `model.pkl`, `scaler.pkl`, `num_cols.pkl`, `feature_columns.pkl` in the `model/` folder.

### Step 4 — Start the web app
```bash
python app.py
```

### Step 5 — Open in browser
```
http://localhost:5000
```

---

## 🔗 API Endpoints

| Method | Endpoint   | Description                     |
|--------|------------|---------------------------------|
| GET    | `/`        | Serves the frontend UI          |
| POST   | `/predict` | Returns fraud prediction (JSON) |
| GET    | `/health`  | Server health check             |

### POST `/predict` — Example Request
```json
{
  "months_as_customer": 200,
  "policy_deductable": 1000,
  "capital_gains": 0,
  "capital_loss": 0,
  "insured_education_level": "High School",
  "insured_relationship": "self",
  "incident_type": "Multi-vehicle Collision",
  "incident_severity": "Major Damage",
  "incident_hour": 14,
  "vehicles_involved": 2,
  "police_report": "YES",
  "property_damage": "NO",
  "bodily_injuries": 1,
  "witnesses": 2,
  "injury_claim": 5000,
  "property_claim": 2000,
  "vehicle_claim": 10000,
  "premium": 1200
}
```

### POST `/predict` — Example Response
```json
{
  "success": true,
  "is_fraud": false,
  "verdict": "CLAIM APPEARS LEGITIMATE",
  "fraud_prob": 18.3,
  "legit_prob": 81.7,
  "top_features": [
    {"feature": "injury_claim", "importance": 14.2},
    {"feature": "vehicle_claim", "importance": 12.8},
    ...
  ]
}
```

---

## 🧠 ML Model Details

- **Algorithm:** Random Forest (100 trees)
- **Target:** `fraud_reported` — Y (Fraud) / N (Legitimate)
- **Class balancing:** `class_weight='balanced'`
- **Preprocessing:** StandardScaler on numerical features, one-hot encoding on categorical
- **Key features:** injury_claim, vehicle_claim, bodily_injuries, witnesses, police_report_available, incident_severity

---

## ⚙️ Tech Stack

| Layer     | Technology              |
|-----------|-------------------------|
| Frontend  | HTML, CSS, Vanilla JS   |
| Backend   | Python Flask            |
| ML Model  | Scikit-learn (Random Forest) |
| AI Report | Claude API (claude-sonnet-4) |
