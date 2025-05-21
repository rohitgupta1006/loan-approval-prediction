from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

class LoanRequest(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: int

app = FastAPI()

model = joblib.load("loan_model.pkl")  
@app.post("/predict")
def predict(data: LoanRequest):
    input_data = [[
        data.Gender,
        data.Married,
        data.Dependents,
        data.Education,
        data.Self_Employed,
        data.ApplicantIncome,
        data.CoapplicantIncome,
        data.LoanAmount,
        data.Loan_Amount_Term,
        data.Credit_History,
        data.Property_Area
    ]]
    
    prediction = model.predict(input_data)[0]
    
    result = "Loan Approved" if prediction == 1 else "Loan Rejected"
    return {"prediction": result}

