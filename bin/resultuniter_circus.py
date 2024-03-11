


import pandas as pd
import sys
import json



df1 = pd.read_csv(sys.argv[1], encoding='shift_jis')
df2 = pd.read_csv(sys.argv[2], encoding='shift_jis')



d = {"results": {"L1": df1.to_dict(), "L3": df2.to_dict()}}

print(d)

with open(sys.argv[3], "wt") as fp:
    fp.write(json.dumps(d, indent=2))


