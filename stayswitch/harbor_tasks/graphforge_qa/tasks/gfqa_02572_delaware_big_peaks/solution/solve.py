import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
rows=list(csv.reader(open(D+"/nwis_peak_delaware_trenton.csv")))[2:]
print(sum(1 for r in rows if len(r)>4 and r[2]>="1950-01-01" and r[4].strip() and float(r[4])>100000))