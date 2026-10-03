import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
r=list(csv.DictReader(open(D+"/eva_spend_by_nigp_commodity_2024.csv")))
c=[x for x in r if float(x["order_count"])>=1000]
b=max(c,key=lambda x:float(x["total_spend"])/float(x["order_count"]))
print(b["NIGP #"])