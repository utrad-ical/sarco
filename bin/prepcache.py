"""
On-disk cache of per-case preprocessing results.

Reading a volume, resampling it and removing the arms dominate the run time,
and the same work is repeated many times: L1 and L3 run the same steps on the
same volume, every training run prepares the same cases again, and a rerun of a
batch starts from scratch. Each step therefore stores what it derived from one
case under a key made of

    - the step name and its version number (CACHE_VERSIONS below)
    - the absolute path, size and modification time of every input file
    - every setting the result depends on

and reuses it only when all of them match. A changed or replaced input file
gets a new key, so a stale entry is never read; it is simply left behind.

When the code of a cached step changes what it produces, bump its version.

cache_dir None disables caching: compute() is called and nothing is written.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np



# bump an entry when the corresponding preprocessing changes its output
CACHE_VERSIONS = {
    "lumbar_gray": 1,       # lumbar.py: projections of the elongated volume
    "lumbar_label": 1,      # lumbar.py: label projections and true z
    "modifier_label": 2,    # lumbarmodifier.py: per-slice label presence (2: nearest-neighbour label resampling)
    "modifier_gray": 1,     # lumbarmodifier.py: search window
    "seg_slice": 1,         # sarcosegmentation.py: the slice to segment
    "trunk_mask": 1,        # extract_trunk.py: body trunk mask (arm removal)
    "label_nonempty": 1,    # readfiles.py: whether a label file has the label
}



def file_signature(path) -> list:
    """
    Identity of an input file: absolute path, size, modification time. For a
    MetaImage header (.mhd) the data file it points to is included as well,
    since the voxels can change while the header stays the same.
    """
    path = Path(str(path).strip()).resolve()
    st = os.stat(path)
    sig = [str(path), int(st.st_size), int(st.st_mtime_ns)]
    if path.suffix.lower() == ".mhd":
        data_file = _mhd_data_file(path)
        if data_file is not None:
            sig.append(file_signature(data_file))
    return sig



def _mhd_data_file(path : Path):
    """The ElementDataFile of a MetaImage header, or None if LOCAL / a list."""
    with open(path, "rt", errors="replace") as f:
        for line in f:
            key, _, value = line.partition("=")
            if key.strip() == "ElementDataFile":
                value = value.strip()
                if value in ("LOCAL", "LIST") or " " in value or "%" in value:
                    return None
                return path.parent / value
    return None



def entry_path(cache_dir, kind : str, key : dict) -> Path:
    payload = json.dumps({"kind": kind, "version": CACHE_VERSIONS[kind], **key},
                         sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return Path(cache_dir) / kind / (digest + ".npz")



def cached(cache_dir, kind : str, key : dict, compute) -> dict:
    """
    compute() returns a dict of numpy arrays or scalars. On a hit the stored
    dict is returned instead (scalars come back as 0-d arrays). A file that
    cannot be read, e.g. left half-written by a killed run, counts as a miss.
    """
    if cache_dir is None:
        return compute()

    path = entry_path(cache_dir, kind, key)
    if path.exists():
        try:
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        except Exception as e:
            print(f"[cache] unreadable entry {path}, recomputing: {e}")

    result = compute()

    # write to a temporary name and rename, so that a concurrent reader or an
    # interrupted run never sees a partial file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **result)
    os.replace(tmp, path)
    return result
