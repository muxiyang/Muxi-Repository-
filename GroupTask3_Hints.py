#Additional loops for section 
ids=df_allloc['ID']
tariffs=[v for v in pd.unique(df_alloc['tariff'])if v !='E']
stimuli=[v for v in pd.unique(df_alloc['stimulus'])if v !='E']

EE=np.random.choice(ids[df_alloc['tariff']=='E'],300,false)

for i in tariffs:
    for j in stimuli:
        n=150 if i=='A' else 50
        temp=np.random.choice(ids[(df_alloc['tarrif']==i)&(df_alloc['stimulus']==j)],n,false)
        EE=np.hstack((EE,temp))