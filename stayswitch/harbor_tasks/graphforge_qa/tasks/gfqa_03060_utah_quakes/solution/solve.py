import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
r=list(csv.DictReader(open(D+"/utah_seismic_events_1980_1984.csv")))
v=[float(x["mag"]) for x in r if 1981<=int(x["time"][:4])<=1983 and x["type"]=="earthquake" and float(x["mag"])>=3.5 and float(x["depth"])>=5.0]
print(round(sum(v)/len(v),2))