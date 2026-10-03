import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
a={r["REPDTE"]:float(r["ROA"]) for r in csv.DictReader(open(D+"/pnc_bank_financials_2021_2026.csv"))}
b={r["REPDTE"]:float(r["ROA"]) for r in csv.DictReader(open(D+"/dsrm_bank_financials.csv"))}
print(sum(1 for k in a if k in b and b[k]>a[k]))