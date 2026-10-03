import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
c=[float(r["ASSET"]) for r in csv.DictReader(open(D+"/cross_river_bank_financials.csv"))]
p=sorted(csv.DictReader(open(D+"/piermont_bank_ubpr_computed_ratios.csv")),key=lambda r:r["REPDTE"])
print(round(max(c)/float(p[0]["TOTAL_ASSETS"]),3))