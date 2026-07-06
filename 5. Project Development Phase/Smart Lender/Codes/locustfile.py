from locust import HttpUser, task, between

class SmartLenderUser(HttpUser):
    # Simulate a user waiting between 1 to 3 seconds between actions
    wait_time = between(1, 3)

    @task(3)
    def view_home(self):
        """Simulates visiting the home page dashboard"""
        self.client.get("/")

    @task(2)
    def view_predict_form(self):
        """Simulates opening the 3-step predict form"""
        self.client.get("/predict")

    @task(1)
    def submit_loan_application(self):
        """Simulates submitting a loan application form"""
        self.client.post("/predict", data={
            "user_name": "Test Runner",
            "user_email": "runner@test.com",
            "user_role": "Applicant",
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "6000",
            "coapplicant_income": "1500",
            "loan_amount": "200",
            "loan_term": "360",
            "credit_history": "1.0",
            "property_area": "Urban"
        })
