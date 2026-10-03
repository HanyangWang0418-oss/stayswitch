import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
p=[float(r["ROE_pct"]) for r in csv.DictReader(open(D+"/piermont_bank_ubpr_computed_ratios.csv")) if r["REPDTE"].startswith("2024")]
c=[float(r["ROE"]) for r in csv.DictReader(open(D+"/cross_river_bank_financials.csv")) if r["REPDTE"].startswith("2014")]
print(round(sum(c)/len(c)-sum(p)/len(p),2))