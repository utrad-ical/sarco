


import pandas as pd
import sys
import json
import os



def records(path):
    """rows of a result table; none when a level could not be processed at all"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[resultuniter] {path} not found; that level is left empty")
        return []
    return pd.read_csv(path, encoding='shift_jis').to_dict(orient='records')



d = {"results": {"L1": records(sys.argv[1]), "L3": records(sys.argv[2])}}

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


