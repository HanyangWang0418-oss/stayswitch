import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
rows=list(csv.reader(open(D+"/nwis_peak_delaware_trenton.csv")))[2:]
a=[float(r[4]) for r in rows if len(r)>4 and r[4].strip() and 1900<=int(r[2][:4])<=1949]
b=[float(r[4]) for r in rows if len(r)>4 and r[4].strip() and 1974<=int(r[2][:4])<=2023]
print(round((sum(b)/len(b)/(sum(a)/len(a))-1)*100,2))