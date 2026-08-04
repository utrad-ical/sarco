


import sys
import numpy as np
import pandas as pd



df_longest = pd.read_csv(sys.argv[1], encoding='shift_jis')

df = pd.read_csv(sys.argv[2], encoding='shift_jis')





zs_estimated = []
zs_true = []
likelihoods = []
zs_modified = []
likelihoods_modified = []


for _, child in df.iterrows():
    
    studyUID = child["{studyUID}"]
    
    mother = df_longest[df_longest["{studyUID}"] == studyUID]
    print(mother)
    if len(mother) == 0:
        # no mother in df_longest
        # kill
        zs_modified.append(-1)
        zs_estimated.append(-1)
        zs_true.append(-1)
        likelihoods.append(0)
        likelihoods_modified.append(0)
        continue
        
    assert len(mother) == 1
    mother_tailmost = mother["{tailmostslicez}"].iloc[0]
    mother_headmost = mother["{headmostslicez}"].iloc[0]
    assert mother_tailmost < mother_headmost
    
    # convert z slice position
    
    def convertz(zmother: int) -> int:
        zcoord = -zmother * mother["{thickness}"] + mother["{headmostslicez}"]
        return int(-np.floor((zcoord - child["{headmostslicez}"]) / child["{thickness}"]))
    
    z_modified = convertz(mother["{z_modified}"])
    z_estimated = convertz(mother["{z_estimated}"])
    z_true = -1 if mother["{z_true}"].iloc[0]<0 else convertz(mother["{z_true}"].iloc[0])

    # copy likelihoods
    likelihood = mother["{likelihood}"].iloc[0]
    likelihood_modified = mother["{likelihoods_modified}"].iloc[0]
    
    zs_modified.append(z_modified)
    zs_estimated.append(z_estimated)
    zs_true.append(z_true)
    likelihoods.append(likelihood)
    likelihoods_modified.append(likelihood_modified)
    

df["{z_estimated}"] = zs_estimated
df["{z_true}"] = zs_true
df["{z_modified}"] = zs_modified
df["{likelihoods_modified}"] = likelihoods_modified
df["{likelihood}"] = likelihoods



df.iloc[:,0:].to_csv(sys.argv[3], encoding='shift_jis')



