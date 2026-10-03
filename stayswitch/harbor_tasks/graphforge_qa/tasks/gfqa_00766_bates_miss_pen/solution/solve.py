import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
h={r["KID"]:r for r in csv.DictReader(open(D+"/bates_field_well_headers.csv"))}
v=[]
for t in csv.DictReader(open(D+"/bates_field_tops.csv")):
    if t["FORMATION"]=="Mississippian System" and t["KID"] in h and h[t["KID"]]["STATUS"]=="OIL" and t["TOP_FT"] and h[t["KID"]]["ROTARY_TOTAL_DEPTH"]:
        v.append(float(h[t["KID"]]["ROTARY_TOTAL_DEPTH"])-float(t["TOP_FT"]))
print(round(sum(v)/len(v),2))