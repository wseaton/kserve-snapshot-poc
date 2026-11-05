C=/compat/usr/local/cuda-13.1/compat
until grep -q SERIES3_DONE /work/run3.log; do sleep 5; done
VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm /exp/exp.sh cuint-ca-580 cuinterpose 1
VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm LD_LIBRARY_PATH=$C:$LD_LIBRARY_PATH /exp/exp.sh cuint-ca-131 cuinterpose 1
echo SERIES4_DONE
