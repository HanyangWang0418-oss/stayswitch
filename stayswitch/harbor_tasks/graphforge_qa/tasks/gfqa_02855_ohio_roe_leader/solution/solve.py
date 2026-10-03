import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
r=[x for x in csv.DictReader(open(D+"/peer_ohio_banks_2023q4.csv")) if float(x["ASSET"])<100000 and x["ROE"].strip()]
print(max(r,key=lambda x:float(x["ROE"]))["NAME"])