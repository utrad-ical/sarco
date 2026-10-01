FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install --no-cache-dir numpy scipy scikit-learn scikit-image pandas \
                               matplotlib cython ipython statsmodels msgpack \
                               memory-profiler pydot h5py pydicom luigi tables \
                               opencv-python opencv-contrib-python imageio SimpleITK

# Requirements for sample code of glow
RUN pip install --no-cache-dir toposort
RUN pip install --no-cache-dir einops tqdm

# ct25d-regressor for the age / phase / body-size estimates (bin/estimate.py),
# pinned to a commit (a tarball, so the image needs no git); --no-deps keeps
# the image's torch
ARG CT25D_COMMIT=c48753e34a778ddf9a9b3bf3b9e44df0562a60fd
RUN pip install --no-cache-dir --no-deps \
    "https://github.com/hanaokalog/ct25d-regressor/archive/${CT25D_COMMIT}.tar.gz"

# copy
COPY plugin.json /
COPY ./ /workspace

CMD bash /workspace/processsingle.sh /circus/in /circus/out
