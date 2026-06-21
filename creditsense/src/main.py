from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from train import init_models, get_risk_verdict

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CreditSense API",
    description="Dual-layer fraud detection and credit risk scoring",
    version="1.0.0",
)

# Allow the React frontend to call this API from a different domain/port.
# In production, replace "*" with your actual Vercel deployment URL for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ─────────────────────────────────────────────────────────────
# All fields are user-entered values from the analyst-facing form.
# Defaults of 0.0 are provided so a partially filled form doesn't crash —
# but in practice the frontend should require all fields before submitting.

class BorrowerProfile(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(0.0, description="Credit card usage relative to limit")
    age: float = Field(0.0, description="Borrower age")
    NumberOfTime30_59DaysPastDueNotWorse: float = Field(0.0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(0.0, description="Monthly debt vs income ratio")
    MonthlyIncome: float = Field(0.0, description="Monthly income")
    NumberOfOpenCreditLinesAndLoans: float = Field(0.0)
    NumberOfTimes90DaysLate: float = Field(0.0)
    NumberRealEstateLoansOrLines: float = Field(0.0)
    NumberOfTime60_89DaysPastDueNotWorse: float = Field(0.0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: float = Field(0.0)
    CombinedLate: float = Field(0.0, description="Total late payment history")
    MonthlyDebt: float = Field(0.0, description="Estimated monthly debt amount")
    IncomePerPerson: float = Field(0.0, description="Income per family member")
    LogIncome: float = Field(0.0, description="Income, log scaled")

    class Config:
        populate_by_name = True


class Transaction(BaseModel):
    velocity: float = Field(0.0, description="Transactions by this card in the last hour")
    amount_deviation: float = Field(0.0, description="Z-score of amount vs card's historical average")
    balance_ratio: float = Field(0.0, description="Normalized count of addresses linked to card")


class RiskRequest(BaseModel):
    borrower_profile: BorrowerProfile
    transaction: Transaction


# ── Startup ────────────────────────────────────────────────────────────────────
# Models are trained/loaded once when the server starts, not on every request.

@app.on_event("startup")
def startup_event():
    print("Starting up CreditSense API...")
    init_models()
    print("CreditSense API ready.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Basic health check — confirms the API is running."""
    return {"status": "ok", "service": "CreditSense API"}


@app.post("/predict")
def predict(request: RiskRequest):
    """
    Main endpoint. Takes borrower profile + transaction details,
    returns trust score, alert level, and full explainability breakdown.

    Frontend (React) calls this with a POST request containing both
    borrower_profile and transaction as nested objects.
    """
    try:
        borrower_dict = request.borrower_profile.dict(by_alias=True)
        transaction_dict = request.transaction.dict()

        result = get_risk_verdict(borrower_dict, transaction_dict)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)