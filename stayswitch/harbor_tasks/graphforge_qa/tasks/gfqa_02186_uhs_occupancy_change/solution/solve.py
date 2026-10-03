import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
def occ(f):
    d=json.load(open(D+'/'+f))[0];return float(d['Total Days (V + XVIII + XIX + Unknown)'])/float(d['Total Bed Days Available'])*100
print(round(occ('cms_cost_report_uh_fy2023.json')-occ('cms_cost_report_uh_fy2022.json'),2))
