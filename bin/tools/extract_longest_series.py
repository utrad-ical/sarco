


import sys
import numpy as np
import pandas as pd



df = pd.read_csv(sys.argv[1], encoding='shift_jis')

dfo = None



studyUIDs = df["{studyUID}"].unique()



for studyUID in studyUIDs:

    # extract each study
    series = df[df["{studyUID}"] == studyUID]

    # get z lengthes
    zlengthes = series["{headmostslicez}"] - series["{tailmostslicez}"]

    # get longest series
    idx = np.argmax(np.array(zlengthes))
    
    if dfo is None:
        dfo = series.iloc[idx:idx+1,:]
    else:
        dfo = pd.concat((dfo, series.iloc[idx:idx+1,:]))
    
    print(dfo)



dfo.to_csv(sys.argv[2])



