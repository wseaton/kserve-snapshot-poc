#!/bin/bash
# Check that CPU KV offload still stores and loads on a running server: cache one long prompt,
# push it out of the GPU KV cache with others, send it again and count the bytes loaded back
# from CPU. Needs a GPU KV cache smaller than the prompts together. Prints the count.
set -u
metric() { $NETNS curl -s localhost:8000/metrics | awk -v m="$1" '$1 ~ m && $1 !~ /^#/ {s+=$2} END {printf "%d\n", s}'; }
prompt() { python3 -c "print(' '.join(f'w$1x{i}' for i in range(600)))"; }
ask() { $NETNS curl -s -m 300 localhost:8000/v1/completions -H 'content-type: application/json' \
  -d "{\"model\":\"m\",\"prompt\":\"$(prompt "$1")\",\"max_tokens\":4,\"temperature\":0}" > /dev/null; }
loaded() { metric 'kv_offload_total_bytes_total.*CPU_to_GPU'; }
before=$(loaded)
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do ask $i; done
ask 0
echo $(( $(loaded) - before ))
