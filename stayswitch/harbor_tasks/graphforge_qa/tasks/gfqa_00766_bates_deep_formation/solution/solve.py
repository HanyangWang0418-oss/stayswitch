import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
g={}
for t in csv.DictReader(open(D+"/bates_field_tops.csv")):
    if t["TOP_FT"]: g.setdefault(t["FORMATION"],[]).append(float(t["TOP_FT"]))
g={k:sum(v)/len(v) for k,v in g.items() if len(v)>=4}
print(max(g,key=g.get))