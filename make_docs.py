"""Generate a synthetic invoice set with known ground truth.

Run once:  uv run make_docs.py
Creates:   docs/*.pdf  and  ground_truth.json
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

random.seed(7)  # same set every run — important for evals

VENDORS = [
    ("Northwind Traders", "Seattle, WA"),
    ("Karachi Paper Co.", "Karachi, PK"),
    ("Blue Ridge Logistics", "Denver, CO"),
    ("Meridian Software Ltd", "London, UK"),
    ("Sahara Office Supply", "Dubai, AE"),
    ("Tallgrass Catering", "Austin, TX"),
]
ITEMS = [
    ("A4 paper, 5 reams", 18.50), ("Toner cartridge", 89.00),
    ("Annual license seat", 240.00), ("Freight, 2 pallets", 310.75),
    ("Onsite support hours", 120.00), ("Packaging tape, 24pk", 32.40),
    ("Catering, 30 pax", 465.00), ("Cloud storage, 1TB", 96.00),
]
CURRENCIES = ["USD", "USD", "USD", "GBP", "AED", "PKR"]

OUT = Path("docs")
OUT.mkdir(exist_ok=True)
truth = []

for n in range(1, 19):
    vendor, city = random.choice(VENDORS)
    cur = random.choice(CURRENCIES)
    issued = date(2026, 1, 1) + timedelta(days=random.randint(0, 240))
    terms = random.choice([15, 30, 45, 60])
    due = issued + timedelta(days=terms)
    lines = random.sample(ITEMS, random.randint(1, 4))
    qtys = [random.randint(1, 6) for _ in lines]
    sub = sum(p * q for (_, p), q in zip(lines, qtys))
    tax = round(sub * random.choice([0.0, 0.05, 0.17]), 2)
    total = round(sub + tax, 2)
    inv_no = f"{random.choice(['INV', 'IN', 'B'])}-{2026}{n:04d}"

    style = n % 3  # three different layouts
    path = OUT / f"invoice_{n:02d}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4

    if style == 0:  # header block, left aligned
        c.setFont("Helvetica-Bold", 20); c.drawString(50, h - 60, "INVOICE")
        c.setFont("Helvetica", 10)
        c.drawString(50, h - 85, vendor); c.drawString(50, h - 98, city)
        c.drawString(50, h - 125, f"Invoice No: {inv_no}")
        c.drawString(50, h - 138, f"Date: {issued:%d %b %Y}")
        c.drawString(50, h - 151, f"Due: {due:%d %b %Y}  (net {terms})")
        y = h - 190
    elif style == 1:  # vendor top-right, ref not labelled "invoice no"
        c.setFont("Helvetica-Bold", 14); c.drawRightString(w - 50, h - 55, vendor)
        c.setFont("Helvetica", 9); c.drawRightString(w - 50, h - 70, city)
        c.setFont("Helvetica", 10)
        c.drawString(50, h - 110, f"Ref {inv_no}")
        c.drawString(50, h - 124, f"Issued {issued:%Y-%m-%d} / payable by {due:%Y-%m-%d}")
        y = h - 165
    else:  # terse, dates in US format, total spelled differently
        c.setFont("Courier-Bold", 12); c.drawString(50, h - 55, vendor.upper())
        c.setFont("Courier", 9)
        c.drawString(50, h - 80, f"{inv_no}   {issued:%m/%d/%Y}   TERMS NET {terms}")
        y = h - 120

    c.setFont("Helvetica", 10)
    for (desc, price), q in zip(lines, qtys):
        c.drawString(55, y, f"{q} x {desc}")
        c.drawRightString(w - 55, y, f"{price * q:,.2f}")
        y -= 16

    y -= 10
    c.line(50, y, w - 50, y); y -= 18
    c.drawRightString(w - 120, y, "Subtotal"); c.drawRightString(w - 55, y, f"{sub:,.2f}"); y -= 14
    if tax:
        c.drawRightString(w - 120, y, "Tax"); c.drawRightString(w - 55, y, f"{tax:,.2f}"); y -= 14
    c.setFont("Helvetica-Bold", 11)
    label = "TOTAL DUE" if style != 2 else "AMOUNT PAYABLE"
    c.drawRightString(w - 145, y, label)
    c.drawRightString(w - 55, y, f"{cur} {total:,.2f}")
    c.save()

    truth.append({
        "file": path.name, "invoice_number": inv_no, "vendor": vendor,
        "issue_date": issued.isoformat(), "due_date": due.isoformat(),
        "currency": cur, "subtotal": round(sub, 2), "tax": tax,
        "total": total, "line_item_count": len(lines),
    })

Path("ground_truth.json").write_text(json.dumps(truth, indent=2))
print(f"wrote {len(truth)} PDFs to docs/ and ground_truth.json")