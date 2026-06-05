from pydantic import BaseModel, Field
from typing import List, Optional


class CustomerFeatures(BaseModel):
    """
    Input schema for prediction.
    Adjust feature names to match your training dataset.
    """

    age: int = Field(..., ge=0, le=120)
    income: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., ge=0)
    employment_years: int = Field(..., ge=0)


class RiskPredictionResponse(BaseModel):
    """
    Output schema for prediction result.
    """

    risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_label: Optional[str] = None