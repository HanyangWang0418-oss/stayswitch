import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
def m(f):
    v=[float(r["EQ"])/float(r["ASSET"])*100 for r in csv.DictReader(open(D+"/"+f)) if r["REPDTE"].startswith("2023")]
    return sum(v)/len(v)
print(round(m("dsrm_bank_financials.csv")-m("pnc_bank_financials_2021_2026.csv"),2))