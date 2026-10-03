import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
def cr(f):
    d=json.load(open(D+'/'+f))[0];return float(d['Total Current Assets'])/float(d['Total Current Liabilities'])
print(round((cr('cms_cost_report_uh_fy2023.json')/cr('cms_cost_report_uh_fy2022.json')-1)*100,2))
