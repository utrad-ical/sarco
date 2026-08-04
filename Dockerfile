FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install --no-cache-dir numpy scipy scikit-learn scikit-image pandas \
                               matplotlib cython ipython statsmodels msgpack \
                               memory-profiler pydot h5py pydicom luigi tables \
                               opencv-python opencv-contrib-python imageio SimpleITK

# Requirements for sample code of glow
RUN pip install --no-cache-dir toposort
RUN pip install --no-cache-dir einops tqdm

# copy
COPY plugin.json /
COPY ./ /workspace

CMD bash /workspace/processsingle.sh /circus/in /circus/out
