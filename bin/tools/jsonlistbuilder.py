


import numpy as np
import json
import glob
import sys
import os



if 1<len(sys.argv):
    rootpath=sys.argv[1]
else:
    rootpath='.'



for jsonpath in glob.glob(rootpath+'/**/*.json', recursive=True):

    print("reading ... " + jsonpath, file=sys.stderr)

    if not os.path.isfile(jsonpath):
        continue
        
    jsondir = os.path.dirname(jsonpath)

    caserootdir = os.path.dirname(jsondir)

    
    with open(jsonpath, 'rt') as f:
        try:
            j = json.load(f)
        except Exception as e:
            print(repr(e), file=sys.stderr)
            print(jsonpath)
            pass
    
    caseId = j["caseId"]
    seriesUid = j['revision']['series'][-1]['seriesUid']
    
    grayfilepath = caserootdir+"/"+caseId+"/"+'vol000.mhd'
    labelfilepath = caserootdir+"/"+caseId+"/"+'vol000-label.mhd'
    
    if not os.path.exists(labelfilepath):
        labelfilepath = 'None'
    
    print("%s, %s, %s, %s" % (grayfilepath,labelfilepath, caseId, seriesUid))
    

