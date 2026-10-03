import os, csv, json
D=os.environ.get("DATA_DIR","/app/data")
d=json.load(open(D+"/svb_financials_q4_2022.json"))["data"]
s=sorted((x["data"]["REPDTE"],float(x["data"]["ASSET"])) for x in d)
print(round(max((s[i][1]/s[i-1][1]-1)*100 for i in range(1,len(s))),2))