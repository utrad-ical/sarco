
FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN pip install --no-cache-dir numpy scipy scikit-learn scikit-image pandas \
                               matplotlib cython ipython statsmodels msgpack \
                               memory-profiler pydot h5py pydicom luigi tables \
                               opencv-python opencv-contrib-python imageio SimpleITK

# Requirements for sample code of glow
RUN pip install --no-cache-dir toposort
RUN pip install --no-cache-dir einops tqdm

# copy
COPY ./ /workspace

CMD bash /workspace/processsingle.sh /circus/in /circus/out
