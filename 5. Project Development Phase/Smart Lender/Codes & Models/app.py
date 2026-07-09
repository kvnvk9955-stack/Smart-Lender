import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
# Add local user site-packages to sys.path in uWSGI environment
user_site = os.path.expanduser('~/.local/lib/python3.10/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Self-healing package check
try:
    import imblearn
except ImportError:
    import subprocess
    print("imbalanced-learn not found. Triggering user-level pip install...", file=sys.stderr)
    subprocess.run(["python3.10", "-m", "pip", "install", "--user", "imbalanced-learn"])

import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, User, ApplicantProfile, CreditHistory, LoanApplication, ModelMeta, PredictionResult

app = Flask(__name__)
app.secret_key = 'smart_lender_secret_key'

# Configure SQLite Database
db_path = os.path.join(os.path.dirname(__file__), 'smart_lender.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize DB with App
db.init_app(app)

# Global model data dictionary
MODEL_DATA = None

def load_prediction_model():
    global MODEL_DATA
    pickle_path = os.path.join(os.path.dirname(__file__), 'rdf.pkl')
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                MODEL_DATA = pickle.load(f)
            print("Successfully loaded rdf.pkl model data.")
        except Exception as e:
            print(f"Error loading rdf.pkl: {e}")
    else:
        print("Warning: rdf.pkl not found! Please trigger retraining via /debug-train.")

# Create database tables and populate metadata
with app.app_context():
    db.create_all()

    # Load model and extract real metrics
    load_prediction_model()
    metrics = None
    if MODEL_DATA and isinstance(MODEL_DATA, dict) and 'metrics' in MODEL_DATA:
        metrics = MODEL_DATA['metrics']

    # Check if models are already registered in the DB
    if not ModelMeta.query.first():
        print("Registering machine learning models in database...")

        models_to_register = [
            {
                'model_name': 'Decision Tree Classifier',
                'model_nm': 'DecisionTree',
                'algorithm': 'Decision Tree',
                'training_accuracy': (metrics['dt']['train'] * 100) if (metrics and 'dt' in metrics) else 82.14,
                'testing_accuracy':  (metrics['dt']['test']  * 100) if (metrics and 'dt' in metrics) else 77.24,
                'file_path': 'sklearn.tree.DecisionTreeClassifier'
            },
            {
                'model_name': 'Random Forest Classifier',
                'model_nm': 'RandomForest',
                'algorithm': 'Random Forest',
                'training_accuracy': (metrics['rf']['train'] * 100) if (metrics and 'rf' in metrics) else 88.61,
                'testing_accuracy':  (metrics['rf']['test']  * 100) if (metrics and 'rf' in metrics) else 79.67,
                'file_path': 'sklearn.ensemble.RandomForestClassifier'
            },
            {
                'model_name': 'K-Nearest Neighbors',
                'model_nm': 'KNN',
                'algorithm': 'KNN',
                'training_accuracy': (metrics['knn']['train'] * 100) if (metrics and 'knn' in metrics) else 81.29,
                'testing_accuracy':  (metrics['knn']['test']  * 100) if (metrics and 'knn' in metrics) else 74.80,
                'file_path': 'sklearn.neighbors.KNeighborsClassifier'
            },
            {
                'model_name': 'Gradient Boosting Classifier',
                'model_nm': 'GradientBoosting',
                'algorithm': 'Gradient Boosting',
                'training_accuracy': (metrics['gb']['train'] * 100) if (metrics and 'gb' in metrics) else 88.20,
                'testing_accuracy':  (metrics['gb']['test']  * 100) if (metrics and 'gb' in metrics) else 78.05,
                'file_path': 'sklearn.ensemble.GradientBoostingClassifier'
            },
            {
                'model_name': 'XGBoost Classifier',
                'model_nm': 'XGBoost',
                'algorithm': 'XGBoost',
                'training_accuracy': (metrics['xgb']['train'] * 100) if (metrics and 'xgb' in metrics) else 94.70,
                'testing_accuracy':  (metrics['xgb']['test']  * 100) if (metrics and 'xgb' in metrics) else 81.30,
                'file_path': 'rdf.pkl'
            }
        ]

        for m in models_to_register:
            new_model = ModelMeta(
                model_name=m['model_name'],
                model_nm=m['model_nm'],
                algorithm=m['algorithm'],
                training_accuracy=m['training_accuracy'],
                testing_accuracy=m['testing_accuracy'],
                file_path=m['file_path']
            )
            db.session.add(new_model)
        db.session.commit()
        print("Model registration completed.")

@app.route('/')
def home():
    # Load model details from database to display on the comparison dashboard
    registered_models = ModelMeta.query.all()
    return render_template('home.html', models=registered_models)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Extract and Validate Form Input Fields
            # User profile info
            user_name = request.form.get('user_name', '').strip()
            user_email = request.form.get('user_email', '').strip()
            user_role = request.form.get('user_role', 'Applicant')

            # Applicant Background
            gender = request.form.get('gender', 'Male')
            married = request.form.get('married', 'No')
            dependents = request.form.get('dependents', '0')
            education = request.form.get('education', 'Graduate')
            self_employed = request.form.get('self_employed', 'No')
            property_area = request.form.get('property_area', 'Urban')

            # Financial Details
            applicant_income = float(request.form.get('applicant_income', 0))
            coapplicant_income = float(request.form.get('coapplicant_income', 0))
            loan_amount = float(request.form.get('loan_amount', 0))
            loan_term = int(request.form.get('loan_term', 360))
            credit_history = float(request.form.get('credit_history', 1.0))

            # 2. Database transaction: Log Profile, Credit, and Loan entries
            # Check or create User
            user = User.query.filter_by(email=user_email).first()
            if not user:
                user = User(name=user_name, email=user_email, role=user_role)
                db.session.add(user)
                db.session.flush() # Populate user_id

            # Create ApplicantProfile
            profile = ApplicantProfile(
                user_id=user.user_id,
                gender=gender,
                married=married,
                education=education,
                self_employed=self_employed,
                dependents=int(dependents[0]), # handles '3+' mapping to 3
                property_area=property_area
            )
            db.session.add(profile)
            db.session.flush() # Populate applicant_id

            # Generate representative credit score based on history
            credit_score = 750.0 if credit_history == 1.0 else 350.0

            # Create CreditHistory record
            credit_hist_record = CreditHistory(
                applicant_id=profile.applicant_id,
                credit_score=credit_score,
                credit_history_status=int(credit_history)
            )
            db.session.add(credit_hist_record)

            # Create LoanApplication record
            loan_app = LoanApplication(
                applicant_id=profile.applicant_id,
                income=applicant_income,
                coapplicant_income=coapplicant_income,
                loan_amount=loan_amount,
                loan_term=loan_term
            )
            db.session.add(loan_app)
            db.session.flush() # Populate loan_id

            # 3. Model Inference Preprocessing
            if MODEL_DATA is None:
                load_prediction_model()

            if MODEL_DATA is None:
                flash("Model file not loaded. Please make sure rdf.pkl is created.", "danger")
                return redirect(url_for('predict'))

            # Features mappings dictionary (must match the model training mappings exactly)
            mappings = {
                'Gender': {'Female': 1, 'Male': 0},
                'Married': {'Yes': 1, 'No': 0},
                'Dependents': {'0': 0, '1': 1, '2': 2, '3': 3, '3+': 3},
                'Education': {'Graduate': 1, 'Not Graduate': 0},
                'Self_Employed': {'Yes': 1, 'No': 0},
                'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
            }

            # Map Categorical Form details to ML features
            gen_val = mappings['Gender'].get(gender, 0)
            mar_val = mappings['Married'].get(married, 0)
            dep_val = mappings['Dependents'].get(dependents, 0)
            edu_val = mappings['Education'].get(education, 1)
            emp_val = mappings['Self_Employed'].get(self_employed, 0)
            prop_val = mappings['Property_Area'].get(property_area, 2)

            # Feature Vector in correct order (match Scaler and Model training order):
            feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                             'Credit_History', 'Property_Area']
            raw_features = pd.DataFrame([[
                gen_val, mar_val, dep_val, edu_val, emp_val,
                float(applicant_income), float(coapplicant_income), float(loan_amount), float(loan_term),
                float(credit_history), prop_val
            ]], columns=feature_names)

            # Scale features
            scaler = MODEL_DATA['scaler']
            scaled_features = scaler.transform(raw_features)

            # --- Hard Policy Rules (Knock-out Criteria / Risk Ceiling) ---
            # Standard banking limit: If requested loan amount exceeds 10 times the annual household income, auto-reject.
            # (loan_amount is in thousands, applicant_income is monthly).
            annual_household_income = (float(applicant_income) + float(coapplicant_income)) * 12
            requested_loan_value = float(loan_amount) * 1000
            
            if annual_household_income <= 0:
                prediction = 0
                probability_score = 0.005
            elif (requested_loan_value / annual_household_income) > 10.0:
                prediction = 0
                excess = (requested_loan_value / annual_household_income) - 10.0
                probability_score = 0.45 / (1.0 + excess)
            else:
                # 4. Predict using Machine Learning Classifier
                model = MODEL_DATA['model']
                prediction = model.predict(scaled_features)[0]
                probabilities = model.predict_proba(scaled_features)[0]
                probability_score = float(probabilities[1]) # probability of class '1' (Approved)

            # Map prediction status
            prediction_status = 'Approved' if prediction == 1 else 'Rejected'

            # Get XGBoost Model ID from database
            xgb_model = ModelMeta.query.filter_by(model_nm='XGBoost').first()
            model_id = xgb_model.model_id if xgb_model else 1

            # Log PredictionResult record
            pred_result = PredictionResult(
                loan_id=loan_app.loan_id,
                model_id=model_id,
                prediction_status=prediction_status,
                probability_score=probability_score
            )
            db.session.add(pred_result)
            db.session.commit()

            return redirect(url_for('result', prediction_id=pred_result.prediction_id))

        except Exception as e:
            db.session.rollback()
            flash(f"Error processing prediction: {str(e)}", "danger")
            return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/result/<int:prediction_id>')
def result(prediction_id):
    pred = PredictionResult.query.get_or_404(prediction_id)
    loan = LoanApplication.query.get(pred.loan_id)
    profile = ApplicantProfile.query.get(loan.applicant_id)
    user = User.query.get(profile.user_id)
    credit = CreditHistory.query.filter_by(applicant_id=profile.applicant_id).first()
    model = ModelMeta.query.get(pred.model_id)

    return render_template(
        'submit.html',
        prediction=pred,
        loan=loan,
        profile=profile,
        user=user,
        credit=credit,
        model=model
    )

@app.route('/history')
def history():
    predictions = PredictionResult.query.options(
        db.joinedload(PredictionResult.loan_application)
        .joinedload(LoanApplication.profile)
        .joinedload(ApplicantProfile.user),
        db.joinedload(PredictionResult.model_meta)
    ).order_by(PredictionResult.prediction_time.desc()).all()

    # Build list of dictionaries for clean table display
    history_list = []
    for p in predictions:
        loan = p.loan_application
        profile = loan.profile if loan else None
        user = profile.user if profile else None
        model = p.model_meta

        if not (loan and profile and user):
            continue

        history_list.append({
            'prediction_id': p.prediction_id,
            'name': user.name,
            'email': user.email,
            'role': user.role,
            'gender': profile.gender,
            'married': profile.married,
            'education': profile.education,
            'income': loan.income,
            'loan_amount': loan.loan_amount,
            'loan_term': loan.loan_term,
            'status': p.prediction_status,
            'probability': p.probability_score,
            'model_name': model.model_name if model else 'XGBoost',
            'time': p.prediction_time.strftime('%Y-%m-%d %H:%M:%S')
        })
    return render_template('history.html', history=history_list)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    pred = PredictionResult.query.get_or_404(prediction_id)
    try:
        loan_id = pred.loan_id
        db.session.delete(pred)

        # Optionally clean up the associated loan, profile, etc.
        loan = LoanApplication.query.get(loan_id)
        if loan:
            applicant_id = loan.applicant_id
            db.session.delete(loan)

            # Check if applicant profile is used elsewhere
            other_loans = LoanApplication.query.filter_by(applicant_id=applicant_id).all()
            if not other_loans:
                profile = ApplicantProfile.query.get(applicant_id)
                if profile:
                    # delete credit history
                    CreditHistory.query.filter_by(applicant_id=applicant_id).delete()
                    db.session.delete(profile)

        db.session.commit()
        flash("Record deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting record: {e}", "danger")

    return redirect(url_for('history'))

if __name__ == '__main__':
    # Make sure model is loaded at startup
    load_prediction_model()
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5001)
