import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
r=[x for x in csv.DictReader(open(D+"/peer_ohio_banks_2023q4.csv")) if x["NAME"]!="NATIONWIDE TRUST CO FSB"]
print(round(sum(float(x["ROA"])*float(x["ASSET"]) for x in r)/sum(float(x["ASSET"]) for x in r),3))