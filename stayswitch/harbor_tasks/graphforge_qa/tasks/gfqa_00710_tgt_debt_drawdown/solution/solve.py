import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
d=json.load(open(D+"/tgt_longterm_debt.json"))["units"]["USD"]
m={x["end"]:x["val"] for x in d}
hi,lo=max(m.values()),min(m.values())
print(round((lo-hi)/hi*100,2))