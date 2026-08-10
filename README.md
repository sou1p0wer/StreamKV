# StreamKV

Official PyTorch code of "StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression".

## Abstract

Video Large Language Models (Video-LLMs) have demonstrated significant potential in the areas of video captioning, search, and summarization. However, current Video-LLMs still face challenges with long real-world videos. Recent methods have introduced a retrieval mechanism that retrieves query-relevant KV caches for question answering, enhancing the efficiency and accuracy of long real-world videos. However, the compression and retrieval of KV caches are still not fully explored. In this paper, we propose **StreamKV**, a training-free framework that seamlessly equips Video-LLMs with advanced KV cache retrieval and compression. Compared to previous methods that used uniform partitioning, StreamKV dynamically partitions video streams into semantic segments, which better preserves semantic information. For KV cache retrieval, StreamKV calculates a summary vector for each segment to retain segment-level information essential for retrieval. For KV cache compression, StreamKV introduces a guidance prompt designed to capture the key semantic elements within each segment, ensuring only the most informative KV caches are retained for answering questions. Moreover, StreamKV unifies KV cache retrieval and compression within a single module, performing both in a layer-adaptive manner, thereby further improving the effectiveness of streaming video question answering. In summary, our main contributions are as follows:

- We propose StreamKV, a training-free framework that seamlessly equips Video-LLMs with advanced KV cache retrieval and compression.
- To better preserve the semantic continuity of video content, StreamKV adopts a semantic partitioning and summary vector mechanism. This approach facilitates both subsequent compression and retrieval.
- To enable KV cache compression in streaming scenarios, we introduce a guidance prompt to capture key semantic elements within each segment, ensuring essential information is retained even under aggressive compression.
- To further improve KV cache retrieval and compression, we propose a Unified Layer-Adaptive KV Selection Module that allocates the selection budget optimally across transformer layers, maximizing informative content under a fixed total budget.

## Directory Structure

```
.
├── data/streamingbench   converted StreamingBench annotations + the convert script
├── model                 StreamKV integration with Video-LLMs (LLaVA-OV)
├── model_zoo/            pretrained Video-LLM checkpoints (download; empty placeholder)
├── results/              evaluation results (empty placeholder)
└── video_qa              StreamingVQA & OfflineVQA solvers + scorers
```

## Preparation

Our setup: 8x NVIDIA H20 (96GB) GPUs.

- Clone this repo: `git clone https://github.com/sou1p0wer/StreamKV.git`
- Prepare the conda environment: `bash prepare.sh`
- Download pretrained Video-LLMs under `model_zoo/`
  - [llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
  - [llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
  - [llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf)
- Download benchmarks under `data/`
  - **StreamingBench** — clone the official repo into `data/streamingbench/StreamingBench/`, then (re)generate the StreamKV-format annotations from the repo root:
    ```bash
    python data/streamingbench/convert_streambench_stream_to_streamkv.py
    ```
    This reads `data/streamingbench/StreamingBench/src/data/questions_*.json` and writes `data/streamingbench/question_{omni,real,sqa,proactive}_streamkv_online.json` with repo-relative `video_path`s.
- Increase the memory-map limit for processes (needed for offloading KV-Caches): `sudo sysctl -w vm.max_map_count=262144`

## Evaluation

`run_eval.py` dispatches to one `eval_<dataset>()` per dataset, spawning `num_chunks` parallel processes (one GPU each; `llava_ov_72b` uses 4 GPUs/chunk → set `num_chunks = #GPUs // 4`).

- **Models**: `llava_ov_0.5b`, `llava_ov_7b`, `llava_ov_72b`
- **Datasets**: `streamingbench_omni`, `streamingbench_real`, `streamingbench_sqa`, `streamingbench_proactive`

Run an evaluation with the canonical config via the ready wrapper:

```bash
bash run_eval.sh
```

## Citation

```bibtex
@misc{chen2025streamkvstreamingvideoquestionanswering,
      title={StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression},
      author={Yilong Chen and Xiang Bai and Zhibin Wang and Chengyu Bai and Yuhan Dai and Ming Lu and Shanghang Zhang},
      year={2025},
      eprint={2511.07278},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.07278}
}
```

## Acknowledgements

Our code is based on [ReKV](https://github.com/Becomebright/ReKV).
