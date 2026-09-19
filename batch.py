import json
import time
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

truth = {r["file"]: r for r in json.load(open("ground_truth.json"))}
results = []

for name in sorted(truth):
    with pdfplumber.open(f"docs/{name}") as pdf:
        text = pdf.pages[0].extract_text()

    try:
        out = generate(
            contents=(
                "Extract the invoice fields.\n"
                "Numeric dates on these documents are US format: MM/DD/YYYY.\n"
                "Return all dates as YYYY-MM-DD.\n\n"
                f"{text}"
            ),
            config={"response_mime_type": "application/json",
                    "response_schema": Invoice},
        ).parsed
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}")
        continue

    t = truth[name]
    row = {"file": name}
    for field in ["invoice_number", "vendor", "issue_date", "due_date", "currency", "total"]:
        got, want = getattr(out, field), t[field]
        if field == "vendor":
            got_cmp, want_cmp = got.strip().lower(), want.strip().lower()
        else:
            got_cmp, want_cmp = got, want
        row[field] = (got_cmp == want_cmp, got, want)

    assert len(row) == 7, f"scorer built {len(row)} keys, expected 7"

    results.append(row)
    print(f"{name}: {sum(1 for f in row if f != 'file' and row[f][0])}/6")

    time.sleep(4)
json.dump(
    [{k: (v if k == "file" else list(v)) for k, v in r.items()} for r in results],
    open("run_02.json", "w"), indent=2,
)

print("\n--- per field ---")
for field in ["invoice_number", "vendor", "issue_date", "due_date", "currency", "total"]:
    hits = sum(1 for r in results if r[field][0])
    print(f"{field:16} {hits}/{len(results)}")