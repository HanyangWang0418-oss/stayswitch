import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
m=json.load(open(D+"/pvgis_austin_tx_production.json"))["outputs"]["monthly"]["fixed"]
s=[x for x in m if x["H(i)_m"]>150]
print(round(sum(x["E_m"] for x in s)/sum(x["H(i)_m"] for x in s),4))