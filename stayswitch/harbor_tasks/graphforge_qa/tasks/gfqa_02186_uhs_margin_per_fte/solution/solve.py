import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
def f(n):
    d=json.load(open(D+'/'+n))[0];return float(d['Net Patient Revenue'])/float(d['FTE - Employees on Payroll'])
print(round(f('cms_cost_report_uh_fy2023.json')-f('cms_cost_report_uh_fy2022.json')))
