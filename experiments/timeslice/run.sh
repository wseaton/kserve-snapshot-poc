#!/bin/bash
# Two jobs take turns on one GPU through llm-d-rl-time-slicing's TimeSlice Orchestrator, with
# snapshotd as the node agent it calls.
#
#   experiments/timeslice/run.sh up     install the orchestrator and point it at the daemon
#   experiments/timeslice/run.sh jobs   run two stand-in jobs and print what happened
#   experiments/timeslice/run.sh verl   start the project's own example: two verl RL jobs whose
#                                       trainers share NODE's GPU (GROUP=trainers; it also takes
#                                       a GPU on two other nodes for the rollout engines)
#   experiments/timeslice/run.sh down   remove everything the other commands made
#
# The daemon on NODE has to serve the agent API: SNAPSHOTD_TIMESLICE_LISTEN=0.0.0.0:9001. The
# orchestrator dials an agent at <node name>:9001, and node names do not resolve in most
# clusters' DNS, so `up` makes a headless Service of that name in the orchestrator's namespace.
#
# Environment:
#   CONTEXT    kube context (default: the current one)
#   NODE       GPU node to run on (required)
#   NAMESPACE  where the daemon runs and the jobs go (default weaton-dev)
#   GROUP      the orchestrator's name for the shared GPU (default demo)
#   GIB        device memory each stand-in job holds (default 72: two do not fit on one
#              141 GiB H200)
#   RUN_SECONDS  how long each verl job trains (default 1800)
set -euo pipefail
cd "$(dirname "$0")"
: "${NODE:?}" "${NAMESPACE:=weaton-dev}" "${GROUP:=demo}" "${GIB:=72}"
k() { kubectl ${CONTEXT:+--context "$CONTEXT"} "$@"; }
SYSTEM=timeslice-system

up() {
  k apply -f orchestrator.yaml --dry-run=client -o name > /dev/null
  k create namespace $SYSTEM --dry-run=client -o yaml | k apply -f -
  k apply -f orchestrator.yaml
  k label node "$NODE" "group.timeslice.io/$GROUP=true" --overwrite
  agent=$(k -n "$NAMESPACE" get pods -l app=snapshotd --field-selector "spec.nodeName=$NODE" -o jsonpath='{.items[0].status.podIP}')
  k apply -f - <<YAML
apiVersion: v1
kind: Service
metadata: {name: $NODE, namespace: $SYSTEM}
spec:
  clusterIP: None
  ports: [{name: grpc, port: 9001}]
---
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: $NODE
  namespace: $SYSTEM
  labels: {kubernetes.io/service-name: $NODE}
addressType: IPv4
ports: [{name: grpc, port: 9001, protocol: TCP}]
endpoints: [{addresses: ["$agent"]}]
YAML
  k -n $SYSTEM rollout status deploy/timeslice-timesliceorchestrator --timeout=180s
}

# The pod the scheduler counts the shared GPU against. Prints the GPU's UUID.
holder() {
  k -n "$NAMESPACE" apply -f - > /dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: timeslice-holder
  labels: {app: timeslice-holder}
  annotations: {sidecar.istio.io/inject: "false"}
spec:
  restartPolicy: Never
  nodeSelector: {kubernetes.io/hostname: $NODE}
  containers:
    - name: main
      image: ghcr.io/wseaton/kserve-snapshot-poc/vllm-checkpoint@sha256:b58cb17af4d87569ba5614bff26fb6c8a8d826f56ca05cfc056fc3d106fc4ee9
      command: [sleep, infinity]
      resources:
        requests: {cpu: "1", memory: 1Gi, nvidia.com/gpu: "1"}
        limits: {nvidia.com/gpu: "1"}
YAML
  k -n "$NAMESPACE" wait --for=condition=Ready pod/timeslice-holder --timeout=300s > /dev/null
  k -n "$NAMESPACE" exec timeslice-holder -- nvidia-smi --query-gpu=uuid --format=csv,noheader
}

jobs() {
  gpu=$(holder)
  echo "the shared GPU is $gpu"
  k -n "$NAMESPACE" create configmap timeslice-job --from-file=job.py --dry-run=client -o yaml | k apply -f -
  for job in job-a job-b; do
    sed "s/__GPU__/$gpu/; s/__JOB__/$job/g; s/__GROUP__/$GROUP/g; s/__NODE__/$NODE/; s/__GIB__/$GIB/" job.yaml |
      k -n "$NAMESPACE" apply -f -
  done
  k -n "$NAMESPACE" wait --for=jsonpath='{.status.phase}'=Succeeded pod -l app=timeslice-job --timeout=1500s ||
    k -n "$NAMESPACE" get pods -l app=timeslice-job
  for job in job-a job-b; do k -n "$NAMESPACE" logs "timeslice-$job" | grep -v WARNING; done | sort
  echo "--- the daemon on $NODE"
  agent=$(k -n "$NAMESPACE" get pods -l app=snapshotd --field-selector "spec.nodeName=$NODE" -o jsonpath='{.items[0].metadata.name}')
  k -n "$NAMESPACE" logs "$agent" -c snapshotd | grep "time-slicing"
}

EXAMPLE=https://raw.githubusercontent.com/llm-d-incubation/llm-d-rl-time-slicing/8757acba5a418c350c9c16ff0cb98530381d3d60/guides/rl-frameworks/verl/examples/fully-async

# The example as shipped, except for what this cluster lacks: the head pods' shared DRA claim
# becomes the runtime's device mount, and since such a pod can see every GPU of the node, the
# head script's check for exactly one is relaxed and CUDA is told which one.
verl() {
  gpu=$(holder)
  echo "the shared trainer GPU is $gpu"
  work=$(mktemp -d)
  for file in job-a-math.yaml job-b-code.yaml; do
    curl -sfL "$EXAMPLE/$file" |
      sed -e "s/\\.default\\.svc/.$NAMESPACE.svc/" \
        -e 's/if \[ "\$NVIS" != 1 \]; then/if [ "$NVIS" -lt 1 ]; then/' \
        -e "/name: RUN_SECONDS/{n;s/\"[0-9]*\"/\"${RUN_SECONDS:-1800}\"/;}" > "$work/$file"
  done
  cat > "$work/kustomization.yaml" <<YAML
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: $NAMESPACE
resources: [job-a-math.yaml, job-b-code.yaml]
commonAnnotations:
  sidecar.istio.io/inject: "false"
labels:
  - pairs:
      timeslice-example: verl
patches:
  - target: {kind: Pod, name: "job-[ab]-head"}
    patch: |
      - {op: remove, path: /spec/resourceClaims}
      - {op: remove, path: /spec/containers/0/resources/claims}
      - {op: add, path: /spec/containers/0/env/-, value: {name: CUDA_VISIBLE_DEVICES, value: $gpu}}
      - {op: add, path: /spec/containers/0/volumeMounts/-, value: {name: gpu, mountPath: /var/run/nvidia-container-devices/$gpu}}
      - {op: add, path: /spec/volumes/-, value: {name: gpu, hostPath: {path: /dev/null}}}
YAML
  k apply -k "$work"
}

down() {
  k -n "$NAMESPACE" delete pod,service,configmap -l timeslice-example=verl --ignore-not-found --wait=false
  k -n "$NAMESPACE" delete pod -l app=timeslice-job --ignore-not-found --wait=false
  k -n "$NAMESPACE" delete pod timeslice-holder --ignore-not-found --wait=false
  k -n "$NAMESPACE" delete configmap timeslice-job --ignore-not-found
  k label node "$NODE" "group.timeslice.io/$GROUP-"
  k delete -f orchestrator.yaml --ignore-not-found
  k delete namespace $SYSTEM --ignore-not-found
}

"${1:?up, jobs, verl or down}"
