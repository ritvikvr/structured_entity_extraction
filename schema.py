from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator

class Parameters(BaseModel):
    user_id: str = Field(description="Unique identifier of the customer or user")
    transaction_id: Optional[str] = Field(default=None, description="Transaction reference ID if present or applicable")
    amount: Optional[float] = Field(default=None, description="Monetary amount involved in the request")
    currency: Optional[str] = Field(default=None, description="Currency code, e.g. USD, EUR, GBP, INR")
    reason: str = Field(description="Normalized concise summary of why the action is being requested")

class StructuredOutput(BaseModel):
    action: Literal[
        "process_refund",
        "cancel_subscription",
        "escalate_ticket",
        "update_profile"
    ] = Field(description="Primary intent/action to execute")
    parameters: Parameters
    requires_human_approval: bool = Field(
        default=False,
        description="True if amount > 100, escalate_ticket, or high-risk compliance profile update"
    )

    @model_validator(mode="after")
    def enforce_deterministic_approval_rules(self) -> "StructuredOutput":
        """
        Deterministic Policy Engine:
        - escalate_ticket -> mandatory True
        - process_refund with amount > 100 -> True
        - process_refund with amount <= 100 -> False
        - update_profile with compliance keys (KYC, legal name, VAT) -> True
        - routine cancel_subscription -> False
        """
        p = self.parameters

        if self.action == "escalate_ticket":
            self.requires_human_approval = True
        elif self.action == "process_refund":
            self.requires_human_approval = bool(p.amount is not None and p.amount > 100.0)
        elif self.action == "update_profile":
            compliance_keywords = ["vat", "kyc", "legal name", "tax", "identity"]
            self.requires_human_approval = any(
                k in p.reason.lower() for k in compliance_keywords
            )
        else:
            self.requires_human_approval = False

        return self
