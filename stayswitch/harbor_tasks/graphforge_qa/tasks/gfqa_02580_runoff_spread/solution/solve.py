import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
r=list(csv.DictReader(open(D+"/mtnhome_runoff_deepperc_2008-2020.csv")))
t={}
for x in r:
    if x["station"]=="Glenns Ferry" and x["variable"]=="runoff":
        t[x["year"]]=t.get(x["year"],0)+float(x["tot_vol_af"])
print(round(max(t.values())-min(t.values()),2))