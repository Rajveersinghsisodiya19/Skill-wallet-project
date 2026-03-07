"""
MediGuard — Train ALL 6 Models
Run: python model/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, os

print("=" * 55)
print("  MediGuard — Training All 6 ML Models")
print("=" * 55)

csv_path = os.path.join(os.path.dirname(__file__), '..', 'insurance_claims.csv')
if os.path.exists(csv_path):
    print("Loading real dataset...")
    df = pd.read_csv(csv_path)
    df.replace('?', np.nan, inplace=True)
    df['collision_type']          = df['collision_type'].fillna(df['collision_type'].mode()[0])
    df['property_damage']         = df['property_damage'].fillna(df['property_damage'].mode()[0])
    df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
else:
    print("No CSV found — using synthetic data (2000 records)")
    np.random.seed(42)
    n = 2000
    fm = np.random.rand(n) < 0.25
    df = pd.DataFrame({
        'months_as_customer':          np.random.randint(1,400,n),
        'policy_deductable':           np.random.choice([500,1000,2000],n),
        'umbrella_limit':              np.random.choice([0,1000000,2000000,3000000],n),
        'capital-gains':               np.random.randint(0,100000,n),
        'capital-loss':                np.random.randint(0,100000,n),
        'incident_hour_of_the_day':    np.random.randint(0,24,n),
        'number_of_vehicles_involved': np.random.randint(1,5,n),
        'bodily_injuries':             np.where(fm,np.random.randint(2,5,n),np.random.randint(0,2,n)),
        'witnesses':                   np.where(fm,np.random.randint(0,2,n),np.random.randint(1,5,n)),
        'injury_claim':                np.where(fm,np.random.randint(10000,80000,n),np.random.randint(500,15000,n)),
        'property_claim':              np.where(fm,np.random.randint(5000,40000,n),np.random.randint(200,8000,n)),
        'vehicle_claim':               np.where(fm,np.random.randint(20000,100000,n),np.random.randint(1000,20000,n)),
        'policy_annual_premium':       np.random.uniform(500,2500,n),
        'insured_education_level':     np.random.choice(['MD','PhD','Associate','Masters','High School','College','JD'],n),
        'insured_relationship':        np.random.choice(['self','spouse','child','other-relative','own-child','not-in-family'],n),
        'incident_type':               np.random.choice(['Single Vehicle Collision','Multi-vehicle Collision','Parked Car','Vehicle Theft'],n),
        'incident_severity':           np.where(fm,np.random.choice(['Major Damage','Total Loss'],n),np.random.choice(['Minor Damage','Trivial Damage','Major Damage'],n)),
        'collision_type':              np.random.choice(['Front Collision','Rear Collision','Side Collision'],n),
        'property_damage':             np.where(fm,np.random.choice(['YES','NO'],n,p=[0.7,0.3]),np.random.choice(['YES','NO'],n,p=[0.3,0.7])),
        'police_report_available':     np.where(fm,np.random.choice(['YES','NO'],n,p=[0.3,0.7]),np.random.choice(['YES','NO'],n,p=[0.8,0.2])),
        'insured_occupation':          np.random.choice(['exec-managerial','handlers-cleaners','other-service','prof-specialty','sales'],n),
        'policy_csl':                  np.random.choice(['100/300','250/500','500/1000'],n),
        'insured_sex':                 np.random.choice(['MALE','FEMALE'],n),
        'fraud_reported':              np.where(fm,'Y','N'),
    })

to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
           'incident_date','incident_state','incident_city','insured_hobbies','auto_make',
           'auto_model','auto_year','_c39','age','total_claim_amount']
df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

cat_df  = pd.get_dummies(X.select_dtypes(include='object'), drop_first=True)
num_df  = X.select_dtypes(include=['int64','float64'])
X_final = pd.concat([num_df, cat_df], axis=1)

out_dir = os.path.dirname(__file__)
joblib.dump(list(X_final.columns), os.path.join(out_dir,'feature_columns.pkl'))

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42, stratify=y)

num_cols_to_scale = [c for c in ['months_as_customer','policy_deductable','umbrella_limit',
    'capital-gains','capital-loss','incident_hour_of_the_day','number_of_vehicles_involved',
    'bodily_injuries','witnesses','injury_claim','property_claim','vehicle_claim'] if c in X_train.columns]

scaler = StandardScaler()
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[num_cols_to_scale] = scaler.fit_transform(X_train_s[num_cols_to_scale])
X_test_s[num_cols_to_scale]  = scaler.transform(X_test_s[num_cols_to_scale])

joblib.dump(scaler,            os.path.join(out_dir,'scaler.pkl'))
joblib.dump(num_cols_to_scale, os.path.join(out_dir,'num_cols.pkl'))

models_def = {
    'decision_tree':       DecisionTreeClassifier(random_state=42),
    'random_forest':       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    'knn':                 KNeighborsClassifier(n_neighbors=5),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'naive_bayes':         GaussianNB(),
    'svm':                 SVC(probability=True, random_state=42),
}

print("\nTraining all 6 models...\n")
results = {}
for name, model in models_def.items():
    print(f"  {name}...", end=' ', flush=True)
    model.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, model.predict(X_train_s))
    te = accuracy_score(y_test,  model.predict(X_test_s))
    results[name] = {'train_acc': round(tr*100,2), 'test_acc': round(te*100,2)}
    joblib.dump(model, os.path.join(out_dir, f'{name}.pkl'))
    print(f"Train {tr*100:.1f}%  Test {te*100:.1f}%  ✅")

joblib.dump(results, os.path.join(out_dir,'model_results.pkl'))
print("\nAll 6 models saved! Run: python app.py")
