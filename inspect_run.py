import json

rows = json.load(open("run_01.json"))
row = next(r for r in rows if r["file"] == "invoice_11.pdf")

for field, val in row.items():
    if field == "file":
        continue
    ok, got, want = val
    mark = "ok  " if ok else "MISS"
    print(f"{field:16} {mark}  got={got!r}  want={want!r}")