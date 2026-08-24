# MultiLayerGNN

Make sure to install Pytorch through

```
MAX_JOBS=12 CUDA_HOME=/usr/local/cuda-13.3 conda run -n torch --no-capture-output pip install torch-scatter --no-build-isolation --no-cache-dir 2>&1
MAX_JOBS=12 CUDA_HOME=/usr/local/cuda-13.3 conda run -n torch --no-capture-output pip install torch-sparse --no-build-isolation --no-cache-dir 2>&1 
MAX_JOBS=12 CUDA_HOME=/usr/local/cuda-13.3 conda run -n torch --no-capture-output pip install lightning --no-build-isolation --no-cache-dir  2>&1
MAX_JOBS=12 CUDA_HOME=/usr/local/cuda-13.3 conda run -n torch --no-capture-output pip install cugraph-cu13 --extra-index-url=https://pypi.nvidia.com --no-build-isolation --no-cache-dir 2>&1
```
