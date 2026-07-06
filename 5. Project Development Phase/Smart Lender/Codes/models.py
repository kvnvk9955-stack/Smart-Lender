from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import event
from sqlalchemy.engine import Engine

db = SQLAlchemy()

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=MEMORY")
        cursor.execute("PRAGMA synchronous=OFF")
        cursor.execute("PRAGMA cache_size=-2000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()
        import sys
        print("SQLite PRAGMA tuning applied successfully!", file=sys.stderr)
    except Exception as e:
        import sys
        print(f"Error applying SQLite PRAGMAs: {e}", file=sys.stderr)


class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    role = db.Column(db.String(50), nullable=False)  # 'Credit Officer', 'Financial Analyst', 'Applicant'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    profiles = db.relationship('ApplicantProfile', backref='user', lazy=True)

class ApplicantProfile(db.Model):
    __tablename__ = 'applicant_profile'
    applicant_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    married = db.Column(db.String(10), nullable=False)
    education = db.Column(db.String(20), nullable=False)
    self_employed = db.Column(db.String(10), nullable=False)
    dependents = db.Column(db.Integer, nullable=False)
    property_area = db.Column(db.String(20), nullable=False)

    # Relationships
    credit_histories = db.relationship('CreditHistory', backref='profile', lazy=True)
    loan_applications = db.relationship('LoanApplication', backref='profile', lazy=True)

class CreditHistory(db.Model):
    __tablename__ = 'credit_history'
    credit_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    applicant_id = db.Column(db.Integer, db.ForeignKey('applicant_profile.applicant_id'), nullable=False)
    credit_score = db.Column(db.Float, nullable=False)
    credit_history_status = db.Column(db.Integer, nullable=False)  # 1 or 0

class LoanApplication(db.Model):
    __tablename__ = 'loan_application'
    loan_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    applicant_id = db.Column(db.Integer, db.ForeignKey('applicant_profile.applicant_id'), nullable=False)
    income = db.Column(db.Float, nullable=False)
    coapplicant_income = db.Column(db.Float, nullable=False)
    loan_amount = db.Column(db.Float, nullable=False)
    loan_term = db.Column(db.Integer, nullable=False)
    application_date = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    prediction_results = db.relationship('PredictionResult', backref='loan_application', lazy=True)

class ModelMeta(db.Model):
    __tablename__ = 'model'
    model_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(100), nullable=False)
    model_nm = db.Column(db.String(100), nullable=False)
    algorithm = db.Column(db.String(100), nullable=False)
    training_accuracy = db.Column(db.Float, nullable=False)
    testing_accuracy = db.Column(db.Float, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    datetime = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    predictions = db.relationship('PredictionResult', backref='model_meta', lazy=True)

class PredictionResult(db.Model):
    __tablename__ = 'prediction_result'
    prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    loan_id = db.Column(db.Integer, db.ForeignKey('loan_application.loan_id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('model.model_id'), nullable=False)
    prediction_status = db.Column(db.String(20), nullable=False)  # 'Approved' or 'Rejected'
    probability_score = db.Column(db.Float, nullable=False)
    prediction_time = db.Column(db.DateTime, default=datetime.utcnow)
