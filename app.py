"""
MediGuard — Health Insurance Fraud Detection
Flask Backend — All 6 Models
Run: python app.py
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib, os

app = Flask(__name__)

BASE = os.path.join(os.path.dirname(__file__), 'model')

# Load shared artifacts
scaler        = joblib.load(os.path.join(BASE, 'scaler.pkl'))
num_cols      = joblib.load(os.path.join(BASE, 'num_cols.pkl'))
feature_cols  = joblib.load(os.path.join(BASE, 'feature_columns.pkl'))
model_results = joblib.load(os.path.join(BASE, 'model_results.pkl'))

# Load all 6 models
MODEL_FILES = {
    'decision_tree':       'Decision Tree',
    'random_forest':       'Random Forest',
    'knn':                 'KNN',
    'logistic_regression': 'Logistic Regression',
    'naive_bayes':         'Naive Bayes',
    'svm':                 'SVM',
}
models = {}
for key in MODEL_FILES:
    models[key] = joblib.load(os.path.join(BASE, f'{key}.pkl'))

print(f"✅ All 6 models loaded | Features: {len(feature_cols)}")


def build_input_df(data):
    raw = {
        'months_as_customer':          float(data.get('months_as_customer', 0)),
        'policy_deductable':           float(data.get('policy_deductable', 1000)),
        'umbrella_limit':              float(data.get('umbrella_limit', 0)),
        'capital-gains':               float(data.get('capital_gains', 0)),
        'capital-loss':                float(data.get('capital_loss', 0)),
        'incident_hour_of_the_day':    float(data.get('incident_hour', 12)),
        'number_of_vehicles_involved': float(data.get('vehicles_involved', 1)),
        'bodily_injuries':             float(data.get('bodily_injuries', 0)),
        'witnesses':                   float(data.get('witnesses', 1)),
        'injury_claim':                float(data.get('injury_claim', 0)),
        'property_claim':              float(data.get('property_claim', 0)),
        'vehicle_claim':               float(data.get('vehicle_claim', 0)),
        'policy_annual_premium':       float(data.get('premium', 1200)),
        'insured_education_level':     data.get('insured_education_level', 'High School'),
        'insured_relationship':        data.get('insured_relationship', 'self'),
        'incident_type':               data.get('incident_type', 'Multi-vehicle Collision'),
        'incident_severity':           data.get('incident_severity', 'Major Damage'),
        'collision_type':              data.get('collision_type', 'Rear Collision'),
        'property_damage':             data.get('property_damage', 'NO'),
        'police_report_available':     data.get('police_report', 'YES'),
        'insured_occupation':          data.get('insured_occupation', 'other-service'),
        'policy_csl':                  data.get('policy_csl', '250/500'),
        'insured_sex':                 data.get('insured_sex', 'MALE'),
    }
    df = pd.DataFrame([raw])
    cat_cols   = df.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_cols]
    cols_to_scale = [c for c in num_cols if c in df_encoded.columns]
    df_encoded[cols_to_scale] = scaler.transform(df_encoded[cols_to_scale])
    return df_encoded


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json(force=True)
        df_input = build_input_df(data)
        all_preds = []

        for key, label in MODEL_FILES.items():
            model      = models[key]
            prediction = model.predict(df_input)[0]
            proba      = model.predict_proba(df_input)[0]
            classes    = list(model.classes_)
            fraud_idx  = classes.index('Y') if 'Y' in classes else 1
            fraud_prob = round(float(proba[fraud_idx]) * 100, 1)
            is_fraud   = prediction == 'Y'

            # Feature importance (only for tree-based models)
            top_features = []
            if hasattr(model, 'feature_importances_'):
                importance  = model.feature_importances_
                top_indices = np.argsort(importance)[::-1][:5]
                top_features = [
                    {"feature": feature_cols[i], "importance": round(float(importance[i]) * 100, 1)}
                    for i in top_indices
                ]

            all_preds.append({
                "key":        key,
                "label":      label,
                "is_fraud":   is_fraud,
                "verdict":    "FRAUD" if is_fraud else "LEGITIMATE",
                "fraud_prob": fraud_prob,
                "legit_prob": round(100 - fraud_prob, 1),
                "train_acc":  model_results[key]['train_acc'],
                "test_acc":   model_results[key]['test_acc'],
                "top_features": top_features,
            })

        # Overall consensus
        fraud_votes = sum(1 for p in all_preds if p['is_fraud'])
        avg_fraud   = round(sum(p['fraud_prob'] for p in all_preds) / len(all_preds), 1)
        consensus   = fraud_votes >= 4  # majority

        return jsonify({
            "success":      True,
            "models":       all_preds,
            "fraud_votes":  fraud_votes,
            "total_models": len(all_preds),
            "avg_fraud_prob": avg_fraud,
            "consensus_fraud": consensus,
            "consensus_verdict": "FRAUD DETECTED" if consensus else "CLAIM APPEARS LEGITIMATE",
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health')
def health():
    return jsonify({"status": "ok", "models_loaded": len(models), "features": len(feature_cols)})


if __name__ == '__main__':
    print("\n🛡️  MediGuard Server Starting...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
