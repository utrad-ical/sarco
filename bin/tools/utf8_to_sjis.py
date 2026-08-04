


import sys

import pandas as pd



df = pd.read_csv(sys.argv[1], encoding='utf-8')



df.to_csv(sys.argv[2], encoding='shift_jis')



