import json
from pydantic import BaseModel
import pdfplumber

from llm import generate

class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    issue_date: str
    due_date: str
    currency: str
    total: float
    payment_method: str | None

target = "invoice_02.pdf"

with pdfplumber.open(f"docs/{target}") as pdf:
    text = pdf.pages[0].extract_text()

print("--- raw text ---")
print(text)

response = generate(
    contents=f"Extract the invoice fields. Dates as YYYY-MM-DD.\n\n{text}",
    config={
        "response_mime_type": "application/json",
        "response_schema": Invoice,
    },
)

print("\n--- model ---")
print(response.parsed)

truth = {r["file"]: r for r in json.load(open("ground_truth.json"))}[target]
print("\n--- truth ---")
print(truth)