import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
d=json.load(open(D+"/streamflow_canyon_creek_13159800.json"))
vals=d["value"]["timeSeries"][0]["values"][0]["value"]
af=sum(float(v["value"]) for v in vals if float(v["value"])>=0)*1.9835
w=list(csv.DictReader(open(D+"/mtnhome_wmis_2023_water_use.csv")))
s=sum(float(r["Volume2023_AF"]) for r in w if r["Source"] in ("CANYON CREEK","LITTLE CANYON CREEK","BENNETT CREEK"))
print(round(s/af*100,2))