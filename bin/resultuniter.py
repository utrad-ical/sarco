


import pandas as pd
import sys



df1 = pd.read_csv(sys.argv[1], encoding='shift_jis')
df2 = pd.read_csv(sys.argv[2], encoding='shift_jis')



dfo = pd.merge(df1, df2, on='id', how='outer')



dfo.iloc[:,1:].to_csv(sys.argv[3], encoding='shift_jis')



