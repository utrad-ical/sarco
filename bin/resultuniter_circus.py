


import pandas as pd
import sys
import json
import os



df1 = pd.read_csv(sys.argv[1], encoding='shift_jis')
df2 = pd.read_csv(sys.argv[2], encoding='shift_jis')



d = {"results": {"L1": df1.to_dict(orient='records'), "L3": df2.to_dict(orient='records')}}

# optional 4th argument: __estimates.csv of estimate.py (age, phase, body size,
# phase-0 equivalent values); shown as results.estimates.0.<name>
if len(sys.argv) > 4 and os.path.exists(sys.argv[4]) and os.path.getsize(sys.argv[4]) > 0:
    try:
        df3 = pd.read_csv(sys.argv[4], encoding='shift_jis')
        d["results"]["estimates"] = df3.to_dict(orient='records')
    except pd.errors.EmptyDataError:
        pass

print(d)

with open(sys.argv[3], "wt") as fp:
    fp.write(json.dumps(d, indent=2))


