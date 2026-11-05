V='import ctypes;l=ctypes.CDLL("libcuda.so.1");l.cuInit(0);v=ctypes.c_int();l.cuDriverGetVersion(ctypes.byref(v));print("cuDriverGetVersion",v.value)'
C=/compat/usr/local/cuda-13.1/compat
python3 -c "$V"
VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm /exp/exp.sh graph-trtllm-580 plain 1 --disable-custom-all-reduce
LD_LIBRARY_PATH=$C:$LD_LIBRARY_PATH python3 -c "$V"
LD_LIBRARY_PATH=$C:$LD_LIBRARY_PATH /exp/exp.sh graph-mnnvl-131 plain 1 --disable-custom-all-reduce
LD_LIBRARY_PATH=$C:$LD_LIBRARY_PATH /exp/exp.sh graph-ca-131 plain 1
echo SERIES3_DONE
