import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
r=list(csv.DictReader(open(D+"/eva_spend_by_nigp_commodity_2024.csv")))
s=sorted((float(x["total_spend"]) for x in r),reverse=True)
print(round(sum(s[:10])/sum(s)*100,2))