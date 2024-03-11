


import sys

import pandas as pd



df = pd.read_csv(sys.argv[1], encoding='shift_jis')



v_n_loc = df.columns.get_loc("{volume_name}")


for i in range(len(df)):

    df.iloc[i, v_n_loc] = sys.argv[2] + df.iloc[i, v_n_loc]



df.iloc[:,1:].to_csv(sys.argv[3], encoding='shift_jis')



