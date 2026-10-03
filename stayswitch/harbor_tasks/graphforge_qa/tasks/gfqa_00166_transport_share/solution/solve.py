import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/ghg_inventory_energy_emissions_table_3-1.csv',header=None,dtype=str)
d=d.iloc[:,1:]
hdr=d.iloc[2].tolist()
d.columns=hdr
d=d.iloc[3:].reset_index(drop=True)
d['Gas/Source']=d['Gas/Source'].astype(str)
v=lambda i,y:float(str(d.loc[i,y]).replace(',',''))
row=lambda lab,start=0:int(d.index[(d['Gas/Source'].str.strip()==lab)&(d.index>=start)][0])
f=row('Fossil Fuel Combustion');t_=row('Transportation')
print(round(100*(v(t_,'2021')/v(f,'2021')-v(t_,'1990')/v(f,'1990')),2))