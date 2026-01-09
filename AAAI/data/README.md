# StreamKV

Core PyTorch code of "StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression"

## File Description

```
.
├── abstract_StreamKV.py          semantic partitioning、summary vector、segment encoding、layer-adaptive allocation module for compression
├── kv_cache_manager.py           memory and retrieval manager for efficient key-value caching、KV compresesion via guidance prompt、KV retrieval via question
├── llava_onevision_StreamKV.py   layer-adaptive allocation module for retrieval、question answering
└── StreamKV_attention.py         StreamKV attention forward function 
```
