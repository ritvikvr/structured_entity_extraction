"""
Dataset generation script for Structured Entity and Intent Extraction.
Generates train.jsonl, valid.jsonl, and test.jsonl with realistic messy,
unstructured customer service conversations, paired with strict JSON schema outputs.
"""

import json
import random
from pathlib import Path
from schema import StructuredOutput, Parameters

SYSTEM_PROMPT = (
    "You are an AI operational dispatch assistant. Your task is to extract structured operational "
    "actions and parameters from messy customer communications. Output strictly valid JSON conforming "
    "to the required schema without any conversational text or markdown code fences."
)

ACTIONS = ["process_refund", "cancel_subscription", "escalate_ticket", "update_profile"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD"]

TEMPLATES = {
    "process_refund": [
        "Hey support, I was billed {currency_sym}{amount} on transaction {tx_id}. My user id is {user_id}. The order arrived completely smashed and unusable! Please get my money back asap.",
        "Yo, refund please. Tx ref: {tx_id}. account id: {user_id}. Charged {currency_sym}{amount} for item I never received after 3 weeks of waiting.",
        "Hi there, can someone reverse the transaction {tx_id}? It was unauthorized, saw {currency_sym}{amount} debited today. My account is {user_id}. Pls refund.",
        "look at tx {tx_id}... i got double charged {currency_sym}{amount} each!! user is {user_id}. Refund the duplicate charge immediately.",
        "Refund request: user {user_id}, transId: {tx_id}. The software subscription renewal charged {currency_sym}{amount} but I cancelled yesterday. I want my money refunded.",
        "I need a refund of {amount} {currency} right now for order/tx {tx_id}! My profile identifier is {user_id}. The service was completely down during our critical launch.",
        "Accidental purchase by my kid: tx #{tx_id}, amt: {currency_sym}{amount}, user tag {user_id}. Please issue a full refund to my card.",
        "tx: {tx_id} | uid: {user_id} | amt: {currency_sym}{amount}. Returning the defective wireless headphones under 30-day warranty. Process refund please."
    ],
    "cancel_subscription": [
        "Cancel my subscription immediately. User ID: {user_id}. I am tired of the constant bugs and price hikes.",
        "Please terminate my recurring billing plan right away. My login id is {user_id}. Switching to a competitor.",
        "Pls cancel my pro plan membership for user account {user_id}. Can no longer afford the monthly fee.",
        "Hi team, don't renew my annual contract. My account code is {user_id}. We are restructuring our team.",
        "Stop auto-renewing my subscription! user_id: {user_id}. The project is complete so I won't be using this tool anymore.",
        "Hey, please downgrade me to the free tier and cancel the premium subscription for {user_id}. Thanks.",
        "I want to unsubscribe from all paid services under account {user_id}. Moving abroad and won't need local cloud storage.",
        "URGENT: Cancel membership for uid {user_id}. Found too many hidden fees in the fine print."
    ],
    "escalate_ticket": [
        "THIS IS UNACCEPTABLE!! I have opened 4 tickets already regarding data loss on account {user_id}. If a senior engineering manager doesn't call me, I will notify our legal counsel!",
        "Escalate ticket immediately! User {user_id}. Production outage affecting 10,000 users right now and your tier 1 support is sending canned responses.",
        "Need supervisor intervention for account {user_id}. Transaction {tx_id} of {currency_sym}{amount} is stuck in limbo and both your agents gave contradicting answers.",
        "I am being locked out of our enterprise workspace {user_id}. Critical security incident suspected. Escalate to your InfoSec team right now!",
        "Your customer service rep was extremely rude and hung up on me. Ticket reference {tx_id}, user is {user_id}. Escalate this to the Head of Support immediately.",
        "Breach of SLA! user {user_id}. Our dedicated instance has been degraded for 48 hours. Escalate this issue to executive management immediately.",
        "Fraud alert on user {user_id}. Multiple suspicious logins from unrecognized IPs attempting to drain wallet funds. Escalate to fraud investigations immediately!",
        "Legal dispute notice regarding contract terms for user {user_id}. Need management review right away."
    ],
    "update_profile": [
        "Hi, please update my registered contact information for user {user_id}. My new email address is updated and I need my mailing address changed to 742 Evergreen Terrace.",
        "User id {user_id} here. I changed my legal name due to marriage and need my KYC identity records updated in your system.",
        "Can you help update the phone number linked to user account {user_id}? I lost my old SIM card and need 2FA switched.",
        "Need to change business billing entity and tax VAT ID for user profile {user_id}.",
        "Profile update request: user {user_id}. Please update emergency contact details and notification preferences.",
        "Hello, my organization rebranded. Update company name and domain on account {user_id}.",
        "Please update delivery address preferences on account {user_id} before the next dispatch cycle.",
        "Changing account recovery email and primary contact person for team account {user_id}."
    ]
}

REASON_TEMPLATES = {
    "process_refund": [
        "Item arrived damaged/defective",
        "Item never received after delivery delay",
        "Unauthorized debit reported by cardholder",
        "Duplicate billing charge",
        "Charged after subscription cancellation",
        "Service downtime during critical period",
        "Accidental purchase by dependent",
        "Product return under warranty policy"
    ],
    "cancel_subscription": [
        "Dissatisfaction with software stability and pricing",
        "Customer migrating to competitor platform",
        "Cost reduction / budget constraints",
        "Customer internal team restructuring",
        "Project completed; service no longer required",
        "Downgrading to free tier",
        "Customer relocating abroad",
        "Dissatisfaction with unexpected fees"
    ],
    "escalate_ticket": [
        "Repeated unresolved tickets with threat of legal action",
        "Critical production outage impacting end users",
        "Conflicting support advice on high-value transaction in limbo",
        "Suspected security incident and enterprise workspace lockout",
        "Customer complaint regarding agent conduct and poor service",
        "Enterprise SLA breach due to prolonged service degradation",
        "Potential account takeover / fraud alert on customer wallet",
        "Legal contract dispute requiring management review"
    ],
    "update_profile": [
        "Update contact email and physical mailing address",
        "Legal name change and KYC record update",
        "Update phone number for two-factor authentication recovery",
        "Update corporate billing entity and VAT tax ID",
        "Update emergency contact and communication preferences",
        "Rebranding company name and domain update",
        "Update default delivery address",
        "Update recovery contact person for account"
    ]
}

def generate_sample(rng: random.Random):
    action = rng.choice(ACTIONS)
    user_id = f"usr_{rng.randint(1000, 99999)}"
    has_tx = action in ["process_refund"] or (action == "escalate_ticket" and rng.random() < 0.4)
    tx_id = f"tx_{rng.randint(100000, 999999)}" if has_tx else None
    
    currency = rng.choice(CURRENCIES) if action == "process_refund" or (action == "escalate_ticket" and has_tx) else None
    currency_sym = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "C$", "AUD": "A$"}.get(currency, "$")
    
    if action == "process_refund":
        amount = round(rng.uniform(15.0, 450.0), 2)
    elif action == "escalate_ticket" and has_tx:
        amount = round(rng.uniform(200.0, 2500.0), 2)
    else:
        amount = None
        
    idx = rng.randint(0, len(TEMPLATES[action]) - 1)
    prompt_template = TEMPLATES[action][idx]
    reason = REASON_TEMPLATES[action][idx]
    
    user_message = prompt_template.format(
        user_id=user_id,
        tx_id=tx_id or f"tx_{rng.randint(100000, 999999)}",
        amount=amount or round(rng.uniform(20.0, 200.0), 2),
        currency=currency or "USD",
        currency_sym=currency_sym
    )
    
    # Noise injection: random typos, lowercasing, prefix text
    if rng.random() < 0.3:
        user_message = user_message.lower()
    if rng.random() < 0.2:
        user_message = f"Fwd: Support ticket notification - {user_message}"
    if rng.random() < 0.15:
        user_message = f"{user_message} Thanks. Sent from my iPhone."

    # Business rule for requires_human_approval:
    # 1. Any escalation always requires human approval
    # 2. Refund > $100 requires human approval
    # 3. Profile update with KYC / Legal name change requires human approval
    if action == "escalate_ticket":
        requires_human_approval = True
    elif action == "process_refund" and (amount is not None and amount > 100.0):
        requires_human_approval = True
    elif action == "update_profile" and ("KYC" in reason or "Legal name" in reason or "VAT" in reason):
        requires_human_approval = True
    else:
        requires_human_approval = False

    structured_data = {
        "action": action,
        "parameters": {
            "user_id": user_id,
            "transaction_id": tx_id,
            "amount": amount,
            "currency": currency,
            "reason": reason
        },
        "requires_human_approval": requires_human_approval
    }
    
    # Validate with Pydantic
    validated = StructuredOutput.model_validate(structured_data)
    json_output = validated.model_dump_json(indent=2)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": json_output}
    ]
    
    return {"messages": messages}

def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(42)
    
    # Generate 500 train samples, 80 validation samples, 80 test samples
    train_data = [generate_sample(rng) for _ in range(500)]
    valid_data = [generate_sample(rng) for _ in range(80)]
    test_data = [generate_sample(rng) for _ in range(80)]
    
    for split, samples in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        path = data_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(samples)} examples to {path}")

if __name__ == "__main__":
    main()
