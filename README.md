# snapshotd

`snapshotd` is a node daemon that brings a vLLM replica up by restoring it from a CRIU checkpoint
instead of cold booting it. It works with a plain `Deployment` as well as KServe's
`LLMInferenceService`, and needs no controller, webhook or custom resource.

**Docs: <https://wseaton.github.io/snapshotd/>**

The fastest path measured leaves the weights out of the checkpoint and pulls them on restore from
a replica that is already serving, over InfiniBand through
[ModelExpress](https://github.com/ai-dynamo/modelexpress), with the checkpoint streamed to the
target node's RAM while it is being taken.

![Seconds until a DeepSeek-V4.1-Flash replica serves on 8 GPUs](docs/static/img/headline.svg)

| Workload | Cold boot | Restore |
| --- | --- | --- |
| DeepSeek-V4.1-Flash, TP=8 + EP, two nodes, weights from a serving replica | 104.7 s | 20.5 s |
| The same, weights from the model volume | 166.1 s | 42.9 to 101.5 s |
| gpt-oss-120b, TP=4, two nodes, 106 GB checkpoint | 72.5 s | 17.2 to 18.6 s |
| Qwen3-8B, TP=4, one node, onto other GPUs | 48.8 s | 8.0 s |

Setup and every other number: [benchmarks](https://wseaton.github.io/snapshotd/benchmarks/).

## Repository layout

| Path | What it is |
| --- | --- |
| [`daemon/`](daemon/README.md) | `snapshotd` itself, and how to build it. |
| [`deploy/`](deploy/README.md) | Kustomize for the daemon, the bench pods and the Dynamo Snapshot path. |
| [`experiments/`](experiments/README.md) | Pod specs and scripts for every measurement, and how to rerun them. |
| [`patches/`](patches/README.md) | Patches to vLLM, ModelExpress, Dynamo Snapshot and CRIU against the commits in [`upstream.tsv`](upstream.tsv). `./checkout.sh vllm` applies them to a clean clone in `src/`. |
| [`docs/`](docs) | The docs site, built with Zola (`zola --root docs serve`). |
| [`kserve-dynamo-snapshot/`](https://github.com/wseaton/kserve-dynamo-snapshot) | Submodule: the earlier KServe integration on Dynamo Snapshot's agent. |

## License

Licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.

Files derived from other projects keep those projects' licenses: the patches under
[`patches/`](patches) (vLLM, ModelExpress and Dynamo Snapshot are Apache-2.0; CRIU is GPL-2.0),
`experiments/weights-file/vllm-overlay.diff`, the schemas under
[`daemon/proto/`](daemon/proto), `experiments/timeslice/orchestrator.yaml`, the Public Sans fonts
under `docs/static/fonts/` (OFL-1.1) and Mermaid under `docs/static/vendor/` (MIT).
