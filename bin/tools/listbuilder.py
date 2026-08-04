
import pandas as pd
import sys
import csv
from pathlib import Path



cases = []
for line in open(sys.argv[1], "rt"):
    cases.append(line.strip(" ").strip("\\").strip("/").strip("\r").strip("\n"))

mother_path = (Path(sys.argv[1]) / "..").resolve()

for desc in ("l1", "l3"):

    records = []

    with open(Path(sys.argv[2]) / Path(f"list_{desc}.txt"), "wt") as fpo:

        for case in cases:

            dict = {}
            dict["{volume_name}"] = (mother_path / Path(f"{case}_gray.nii.gz")).resolve()
            dict["{label_volume_name}"] = (mother_path / Path(f"{case}_{desc}.nii.gz")).resolve()
            records.append(dict)

        pd.DataFrame.from_records(records).to_csv(fpo)
    
    



