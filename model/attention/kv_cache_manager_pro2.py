import math
import torch
from typing import Optional, Tuple
import torch.nn.functional as F
from .dot_production_attention import get_multi_stage_dot_production_attention

import pandas as pd
import json
import os
import csv
'''
**************************************max************************************
对应self.qa_model.encode_init_prompt()之后就self.init_exc=True的做法
好处是处理起来比较统一和直观
坏处是当视频较短时仍按照长视频的处理方式来进行处理
**************************************max************************************
'''

def custom_json_dump(data, file, indent=4):
    """
    自定义 JSON 序列化函数
    - 外层结构正常缩进
    - 最内层列表保持紧凑（不换行）
    """
    def serialize_inner_list(lst):
        """将最内层列表序列化为紧凑字符串（无空格/换行）"""
        return '[' + ','.join(map(str, lst)) + ']'

    # 处理顶层字典
    file.write('{\n')
    first_key = True
    for key, outer_list in data.items():
        if not first_key:
            file.write(',\n')
        first_key = False
        
        # 序列化键
        file.write(f'{" " * indent}{json.dumps(key)}: [')
        
        # 处理外层列表中的每个内层列表
        first_inner = True
        for inner_list in outer_list:
            if not first_inner:
                file.write(',')
            first_inner = False
            
            # 紧凑序列化内层列表
            compact_str = serialize_inner_list(inner_list)
            file.write('\n' + ' ' * (indent * 2) + compact_str)
        
        file.write(f'\n{" " * indent}]')
    file.write('\n}')

# Allocate a fixed-size block of GPU memory specifically for storing the KV-Cache of the local_window.
class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units  # 缓存中单元的数量
        self.unit_size = unit_size  # block_size * hidden_dim * 2  每个单元的大小
        self.dtype = dtype
        self.data = torch.empty(
            (num_units, unit_size),
            device = "cuda",
            dtype=dtype
        ) #创建一个空的CUDA张量
        self.idle_set = set(list(range(num_units))) # 初始化空闲单元集合

    def alloc(self):
        assert len(self.idle_set) > 0 # 确保有空闲单元
        idx = self.idle_set.pop()
        return self.data[idx], idx # 返回单元和单元索引

    def delete(self, idx):
        assert idx not in self.idle_set # 确保单元索引不在空闲单元集合中
        self.idle_set.add(idx)


# The KV-Cache management unit supports data transfer between the CPU and GPU.
class MemoryUnit:
    # Initialize the KV-Cache management unit and store it on the CPU.
    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda: # 如果kv在GPU上
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        if pin_memory: # 如果需要pin memory，固定在内存中
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)
        #不用管
        if load_to_cache: 
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True) # 复制kv[0]到gpu_data[0]
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    # Load data from the CPU to the GPU and copy it to 'target' when necessary.
    # target: 2x (n_head, n_token, head_dim), on GPU
    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None

            return False, target_event
        #if self.gpu_data is None
        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)
        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)

        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    # Get the KV-Cache stored on GPU
    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    # Clear the KV-Cache stored on GPU
    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def calculate_cpu_memory(self):
        return len(self.cpu_data) * self.cpu_data[0].numel() * self.cpu_data[0].element_size()


# A dynamically growing vector cache on the GPU, used to store representative vectors of video frames.
class VectorTensor:
    # Initialize an empty cache of size (16, hidden_dim) on the GPU.
    def __init__(
        self, 
        hidden_size,
        element_dtype,
        device
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    # Double the size of the cache.
    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size,...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    # Append a frame vector to the cache, and expand the cache if it exceeds the current cache size.
    def append(self, tensor: torch.Tensor): # tensor: (append_l, hidden_size)
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size, f'{tensor.size(1)}, {self.hidden_size}'
        assert tensor.is_contiguous()

        append_l = tensor.size(0) #append_len

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)

        self.length += append_l
    
    def delete(self, del_idx: torch.Tensor):
        device = self.data.device
        del_idx = del_idx.to(device)
        all_idx = torch.arange(self.length, device=device)
        mask = ~torch.isin(all_idx, del_idx)
        keep_indices = all_idx[mask]
        num_keep = keep_indices.size(0)
        self.data[:num_keep] = self.data[keep_indices]
        self.length = num_keep

    # Get the cached frame vectors
    def get_data(self):
        return self.data[:self.length, ...]

    def get_chunk(self,st,ed):
        return self.data[st:ed+1, ...]

    def get_mean(self,st,ed):
        return self.data[st:ed+1, ...].mean(dim=0,keepdim=True)

    def get_window_cosine_similarity(self, window_size, tensor: torch.Tensor):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        key = self.data[self.length-window_size:self.length].float() # (window_size, D)
        query = tensor[None, :].float() # (1, D)
        logits = torch.matmul(query, key.T)[0]

        assert logits.dim() == 1 and logits.size(0) == window_size,f'{logits.dim()}, {logits.size(0)},{window_size}'
        return logits

    def get_cosine_similarity(self, tensor: torch.Tensor): # tensor: (D,) 计算tensor和self.data的余弦相似度
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        key = self.data[:self.length].float()  # (T, D), convert to fp32 to prevent numerical overflow
        query = tensor[None, :].float()  # (1, D)

        logits = torch.matmul(query, key.T)[0]  # (T,)
        #print(f'raw_logits: {logits}')
        # ***********L2 normalization***********
        # l2_norm = logits.norm(p=2)
        # normalized_logits = logits / l2_norm
        # ***********Z-score normalization***********
        mean = torch.mean(logits)
        std = torch.std(logits)
        normalized_logits = (logits - mean) / std #* torch.pow(torch.tensor(logits.size(0)),0.25)
        #print(f'normalized_logits: {normalized_logits}')
        logits = F.softmax(normalized_logits, dim=0)
        #print(f'softmax_logits: {logits}')

        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def __len__(self):
        return self.length


GLOBAL_STREAM = None


class ContextManager:
    def __init__(self, 
                 position_embedding,
                 n_init, n_local, 
                 block_size, max_cached_block, topk, chunk_size, exc_block_size, 
                 fattn: bool = False,
                 async_global_stream: bool = False,
                 pin_memory: bool = False,
    ):

        self.length = 0  # number of tokens in the KV-Cache
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local #局部窗口大小
        self.block_size = block_size #等于单帧token数
        self.max_cached_block = max_cached_block # maximum number of blocks in the KV-Cache
        self.max_retrive_block = max_cached_block # maximum number of blocks to retrive
        self.exc_block_size = exc_block_size #encode video frames 的 block size，等于单帧token数
        assert exc_block_size <= n_local # no global token in input
        self.topk = topk # number of blocks to retrieve
        self.chunk_size = chunk_size #分组索引
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn # whether to use fast attention
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream # whether to use async global stream
        self.pin_memory = pin_memory
        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.reset_retrieval()
    # remove the least recently used blocks 从self.global_blocks中offload 只是remove 不会load进新内容
    def _remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks = None): 
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    # handle GQA, k: (batch_size, n_head_kv, length, dim_head) -> (batch_size, n_head, length, dim_head)
    def _from_group_kv(self, tensor):
        # tensor: (batch_size, n_head_kv, length, dim_head)
        assert tensor.dim() == 4 
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_units, self.unit_size_kv, 1, length, dim_head))  # (batch_size, n_head_kv, 1, length, dim_head)
        tensor = tensor.expand((self.num_units, self.unit_size_kv, num_group, length, dim_head)).reshape((self.num_units, self.num_heads, length, dim_head))  # (batch_size, n_head, length, dim_head)
        return tensor
    
    def init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        """
        Only use the metadata of these parameters, such as shape, dtype, and device.
        """
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv

        self.global_blocks = [[] for _ in range(self.num_units)] # context memory's KV-Cache: [ batch_size x [memory_unit] ]
        self.cached_blocks = [{} for _ in range(self.num_units)] # relavency scores of blocks: batch_size x {block_id: block_score}
        self.num_global_block = 0

        # context memory's representative keys: batch_size x (n_blocks, hidden_dim)
        self.block_k = [VectorTensor(
            dim_head * self.unit_size, global_k.dtype, global_k.device
        ) for _ in range(self.num_units)]

        # local KV
        self.local_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)  # (batch_size, n_head_kv, 0, dim_head)
        self.local_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        # global KV that are not yet processed into blocks.
        # 2 x (batch_size, n_head_kv, length, dim_head)
        self.global_remainder = (
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )

        # init KV
        self.init_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        # buffering global KV during attention computations
        # (2, batch_size, n_head_kv, L, dim_head)
        # L = n_init + n_retrieve
        buffer_len = self.max_retrive_block * self.block_size + self.n_init #token level
        self.global_buffer = torch.zeros(
                (2, self.num_units, self.unit_size_kv, buffer_len , dim_head),
                dtype = global_k.dtype, device=global_k.device
            )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype
        )  # (max_cached_block * batch_size, block_size * D * 2)

        #chunk representative keys: batch_size x (n_chunks, hidden_dim)
        self.chunk2block = [[] for _ in range(self.num_units)] #换个思路 通过len_q就能直接知道chunk2block 通过
        self.chunk_k = [VectorTensor(
            dim_head * self.unit_size, global_k.dtype, global_k.device
        ) for _ in range(self.num_units)]
        self.num_global_chunk = 0
        self.retrive_k = [VectorTensor(
            dim_head * self.unit_size, global_k.dtype, global_k.device
        ) for _ in range(self.num_units)]
        self.retrive_threshold = 0.97
        with open('model/attention/retrival_budget.csv','r') as f:
            reader = csv.reader(f)
            self.retrieval_budget = [int(x) for x in list(reader)[0]]
        self.initialized = True

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.similarity = None
        self.retrieved_block_indices = None
        self.to_retrieve = False

    def set_retrieved_block_indices(self, retrieved_block_indices):
        # retrieved_block_indices (list): batch_size x n_frames
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices

    def compress_kv(self, video_len, prompt, compress_ratio):
        num_frames = video_len // self.block_size
        prompt = prompt.mean(dim=2, keepdim=False)
        assert prompt.shape == (self.num_units, self.unit_size, self.dim_head)
        prompt = prompt.reshape(self.num_units, self.dim_head * self.unit_size)
        logits = torch.stack([self.block_k[u].get_window_cosine_similarity(num_frames,prompt[u]) for u in range(self.num_units)]) #(batch_size, num_frames)
        num_frames_after_compressed = max(1, int(num_frames * compress_ratio))
        all_indices = torch.arange(num_frames).to(logits.device)
        kept_indices = logits.topk(num_frames_after_compressed, dim=1).indices
        kept_indices = kept_indices.sort(dim=1)[0] # (batch_size, num_frames_after_compressed)
        evicted_indices = [all_indices[~torch.isin(all_indices, kept_indices[u])] for u in range(self.num_units)]
        evicted_indices = [ekv.sort(descending=True)[0] for ekv in evicted_indices]
        offset = self.num_global_block - num_frames
        evicted_indices = [ekv + offset for ekv in evicted_indices]
        # print('------------before compression----------')
        # print(f'num_global_block: {self.num_global_block}')
        # print(f'num_block_k:{len(self.block_k[0])}')
        # print(f'video_len: {video_len//self.block_size}')
        for u in range(self.num_units):
            for idx in evicted_indices[u]:
                if idx in self.cached_blocks[u].keys():
                    self.global_blocks[u][idx].offload()  #修改global_blocks
                    self.cached_blocks[u].pop(idx)
                del self.global_blocks[u][idx]
                #print(f'evicted_indices: {idx}')
                self.num_global_block -= 1                #修改num_global_block
            self.block_k[u].delete(evicted_indices[u])    #修改block_k
        
        # 更新local_kv
        local_k_s = self.local_k[:,:,:-video_len,:]
        local_v_s = self.local_v[:,:,:-video_len,:]
        local_k_e = self.local_k[:,:,-video_len:,:]
        local_v_e = self.local_v[:,:,-video_len:,:]
        block_size = self.block_size
        batch = kept_indices.size(0)  # batch size
        offsets = torch.arange(block_size, device=kept_indices.device)  # shape:(block_size,)
        token_indices = kept_indices.unsqueeze(-1) * block_size + offsets  # (batch, num_kept_blocks, block_size)
        token_indices = token_indices.reshape(batch, -1)  # (batch, num_tokens_kept)
        token_indices_exp = token_indices[:, None, :, None].expand(-1, local_k_e.shape[1], -1, local_k_e.shape[3])
        local_k_e_kept = torch.gather(local_k_e, 2, token_indices_exp)
        local_v_e_kept = torch.gather(local_v_e, 2, token_indices_exp)
        local_k = torch.cat([local_k_s, local_k_e_kept], dim=2)
        local_v = torch.cat([local_v_s, local_v_e_kept], dim=2)
        if local_k.size(-2) > self.n_local:
            local_k = local_k[:, :, -self.n_local:, :]
            local_v = local_v[:, :, -self.n_local:, :]
        self.local_k = local_k
        self.local_v = local_v
        # print('------------after compression----------')
        # print(f'evicted_num: {len(evicted_indices[0])}')
        # print(f'num_global_block: {self.num_global_block}')
        # print(f'num_block_k:{len(self.block_k[0])}')

        #需要修改的地方：local_kv 将append里面的截断放到这个函数里

    def get_retrieved_kv(self,layer_idx, query=None): #将init KV和retrieved KV加载到global_h_k, global_h_v，没global_buffer什么事
        """retrieve context blocks with retrieved_block_indices
        query: (batch_size, num_heads, seq_len, dim_head)
        return [init_k, retrieved_k] and the respective v
        """

        if query is not None:  # retrieve based on the attention score between query and context's representative keys
            block_topk = self._calc_block_topk(layer_idx, query) # block_topk: batch_size x topk
            self.set_retrieved_block_indices(block_topk)

        assert len(self.retrieved_block_indices) == self.num_units

        buffer_len = block_topk.size(1) * self.block_size + self.n_init
        global_h_k, global_h_v = torch.zeros(
                (2, self.num_units, self.unit_size_kv, buffer_len , self.dim_head),
                dtype = self.global_buffer.dtype, device=self.global_buffer.device
            )

        with torch.cuda.stream(GLOBAL_STREAM):
            # offload LRU blocks
            for u in range(self.num_units):
                num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                for b_idx in self.retrieved_block_indices[u]:
                    if b_idx not in self.cached_blocks[u]:
                        num_remove += 1
                self._remove_lru_blocks(u, num_remove, self.retrieved_block_indices[u]) #给self.retrieved_block_indices腾出空间

            self.load_count += 1
            for u in range(self.num_units):
                for b_idx in self.retrieved_block_indices[u]:
                    self.cached_blocks[u][b_idx] = self.load_count
            
            # no need to load init KV to global_buffer
            init_st = 0
            init_ed = init_st + self.n_init
            global_h_k[:, :, init_st:init_ed] = self.global_buffer[0][:, :, init_st:init_ed]
            global_h_v[:, :, init_st:init_ed] = self.global_buffer[1][:, :, init_st:init_ed]
            ed = init_ed
            assert self.global_buffer_init_st == init_st or self.global_buffer_init_ed == init_ed

            # load retrieved context KV 
            for u in range(self.num_units):
                # assert len(self.retrieved_block_indices[u]) == block_num
                assert self.retrieved_block_indices[u][-1] < self.num_global_block, f'{self.retrieved_block_indices[u][-1]}, {self.num_global_block}'
                for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                    # load global_blocks[u][b_idx] onto GPU and make a copy to (global_h_k, global_h_v)
                    st = init_ed + cnt * self.block_size
                    ed = st + self.block_size
                    self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :])) #reload to GPU and copy it to global_h_k, global_h_v


            # global_h_k = global_h_k[:, :, :ed, :]
            # global_h_v = global_h_v[:, :, :ed, :]
            # assert global_h_k.size(-2) == global_h_v.size(-2) == self.n_init + block_num * self.block_size

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        assert global_h_k.size(-2) == self.n_init + block_topk.size(1) * self.block_size
        return global_h_k, global_h_v 

    def get_indices(self,layer_idx, logits):
        ret = []
        layer_budget = self.retrieval_budget[layer_idx]
        for u in range(self.num_units):
            sorted_logits, sorted_indices = torch.sort(logits[u], descending=True)
            accumulated_sum = 0.0
            selected_indices = []
            for i, logit in enumerate(sorted_logits):
                accumulated_sum += logit.item()
                selected_indices.append(sorted_indices[i].item())
                if i >= self.max_retrive_block-1 or i >= layer_budget-1 or accumulated_sum >= self.retrive_threshold:
                    break
            ret.append(sorted(selected_indices))
        return ret, accumulated_sum

    # Get the indices of the top-k vectors in self.block_k[u] that have the highest similarity with global_h_q[u].
    # ret: batch_size x topk
    def _calc_block_topk(
        self,layer_idx, global_h_q # (batch_size, num_heads, length, dim_head)
    ):
        global_h_q = global_h_q.mean(dim=2, keepdim=False)  # (batch_size, num_heads, dim_head) 用query的mean作为representative token
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(self.num_units, self.dim_head * self.unit_size)  # (batch_size, dim_head * num_heads)
        logits = None

        # #one-stage
        # if self.num_global_block <= 10000:      
        #     if self.num_global_block <= self.retrieval_budget[layer_idx] and self.num_global_block <= self.max_retrive_block:
        #         ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        #         accumulated_sum = 1
        #         #logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #         #print(f'logits:{logits}')
        #     else:
        #         logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #         ret,accumulated_sum = self.get_indices(layer_idx,logits)
        #     print(f'layer{layer_idx},retrive_len{len(ret[0])},accumulated_sum{accumulated_sum}')
        # #two-stage
        # else:        
        #     hash_map = [[] for _ in range(self.num_units)]
        #     retrive_k = [VectorTensor(
        #         self.dim_head * self.unit_size, global_h_q.dtype, global_h_q.device
        #     ) for _ in range(self.num_units)]
        #     chunk_logits = torch.stack([self.chunk_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)]) #(batch_size, num_global_chunk)
        #     retrieved_chunk_indices = self.get_indices(chunk_logits)
        #     for u in range(self.num_units):
        #         for chunk in retrieved_chunk_indices[u]:
        #             chunk_st = self.chunk2block[u][chunk][0] #chunk在global_blocks中的起始索引
        #             chunk_ed = self.chunk2block[u][chunk][-1]#chunk在global_blocks中的结束索引
        #             hash_map[u].extend(self.chunk2block[u][chunk])
        #             retrive_k[u].append(self.block_k[u].get_chunk(chunk_st, chunk_ed))
        #     logits = torch.stack([retrive_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #     ret = self.get_indices(logits)
        #     for u in range(self.num_units):
        #         for idx in range (len(ret[u])):
        #             ret[u][idx] = hash_map[u][ret[u][idx]]

        #---------------------one-stage topk-----------------------
        # block_logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        # if self.num_global_block <= self.topk:
        #     ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        # else:
        #     logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])  # (batch_size, block_num)
        #     ret = logits.topk(self.topk, dim=1).indices #所有chunk块之中的topk//chunk_size (batch_size, topk//chunk_size)
        #     ret = ret.sort(dim=1)[0]
        #     ret = ret.cpu().tolist()
        #---------------------one-stage prefix topk-----------------------
        topk = min(self.retrieval_budget[layer_idx], self.max_retrive_block)
        block_logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        if self.num_global_block <= topk:
            ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        else:
            logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])  # (batch_size, block_num)
            ret = logits.topk(topk, dim=1).indices #所有chunk块之中的topk//chunk_size (batch_size, topk//chunk_size)
            ret = ret.sort(dim=1)[0]
            ret = ret.cpu().tolist()
        #print(f'layer{layer_idx},retrive_len{len(ret[0])}')
        #--------------------one-stage prefix----------------------
        # if self.num_global_block <= self.retrieval_budget[layer_idx] and self.num_global_block <= self.max_retrive_block:
        #     ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        #     accumulated_sum = 1
        #     #logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #     #print(f'logits:{logits}')
        # else:
        #     logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #     ret,accumulated_sum = self.get_indices(layer_idx,logits)
        # print(f'layer{layer_idx},retrive_len{len(ret[0])},accumulated_sum{accumulated_sum}')
        #--------------------two-stage topk------------------------
        # if self.num_global_block <= self.topk:
        #     ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        # else:
        #     min_chunk_size = 8
        #     if self.num_global_chunk <= self.topk // min_chunk_size + 1: #最末尾的chunk可能会长度小于8
        #         logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #         ret = logits.topk(self.topk, dim=1).indices #(batch_size, topk) 
        #         ret = ret.sort(dim=1)[0]
        #         ret = ret.cpu().tolist()
        #     else:
        #         hash_map = [[] for _ in range(self.num_units)]
        #         retrive_k = [VectorTensor(
        #             self.dim_head * self.unit_size, global_h_q.dtype, global_h_q.device
        #         ) for _ in range(self.num_units)]
        #         chunk_logits = torch.stack([self.chunk_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #         retrieved_chunk_indices = chunk_logits.topk(self.topk // min_chunk_size + 1, dim=1).indices #(batch_size, topk//min_chunk_size + 1)
        #         retrieved_chunk_indices = retrieved_chunk_indices.sort(dim=1)[0]
        #         for u in range(self.num_units):
        #             for chunk in retrieved_chunk_indices[u]:
        #                 chunk_st = self.chunk2block[u][chunk][0] #chunk在global_blocks中的起始索引
        #                 chunk_ed = self.chunk2block[u][chunk][-1]#chunk在global_blocks中的结束索引
        #                 hash_map[u].extend(self.chunk2block[u][chunk])
        #                 retrive_k[u].append(self.block_k[u].get_chunk(chunk_st, chunk_ed))
        #         logits = torch.stack([retrive_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #         assert logits.size(1) >= self.topk, f'logits:{logits}'
        #         ret = logits.topk(self.topk, dim=1).indices #(batch_size, topk)
        #         ret = ret.sort(dim=1)[0]
        #         for u in range(self.num_units):
        #             for idx in range (len(ret[u])):
        #                 ret[u][idx] = hash_map[u][ret[u][idx]]
        #         ret = ret.cpu().tolist()
        #--------------------two-stage prefix----------------------
        # if self.num_global_block <= self.max_retrive_block:
        #     ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        # else:
        #     min_chunk_size = 8
        #     if self.num_global_chunk <= self.max_retrive_block // min_chunk_size + 1: #最末尾的chunk可能会长度小于8
        #         logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #         ret = self.get_indices(logits)
        #     else:
        #         hash_map = [[] for _ in range(self.num_units)]
        #         retrive_k = [VectorTensor(
        #             self.dim_head * self.unit_size, global_h_q.dtype, global_h_q.device
        #         ) for _ in range(self.num_units)]
        #         chunk_logits = torch.stack([self.chunk_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)]) #(batch_size, num_global_chunk)
        #         retrieved_chunk_indices = self.get_indices(chunk_logits)
        #         for u in range(self.num_units):
        #             for chunk in retrieved_chunk_indices[u]:
        #                 chunk_st = self.chunk2block[u][chunk][0] #chunk在global_blocks中的起始索引
        #                 chunk_ed = self.chunk2block[u][chunk][-1]#chunk在global_blocks中的结束索引
        #                 hash_map[u].extend(self.chunk2block[u][chunk])
        #                 retrive_k[u].append(self.block_k[u].get_chunk(chunk_st, chunk_ed))
        #         logits = torch.stack([retrive_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
        #         ret = self.get_indices(logits)
        #         for u in range(self.num_units):
        #             for idx in range (len(ret[u])):
        #                 ret[u][idx] = hash_map[u][ret[u][idx]]
        #-------------------json----------------------
        #写入json，缺点每次得重新读取重新全写，不能追加
        # file_path = 'prefix_data/mlvu.json'
        # try:
        #     with open(file_path, 'r') as file:
        #         data = json.load(file)
        # except json.JSONDecodeError:
        #     data = {}
        # data[str(layer_idx)].append(block_logits[0].cpu().tolist())
        # with open(file_path, 'w') as file:
        #     custom_json_dump(data, file, indent=4)
        #-------------------csv----------------------
        #写入csv，能够追加
        # file_path = f'prefix_data/qs_ego4d/{layer_idx}.csv'
        # list_logits = block_logits[0].cpu().tolist()
        # round_logits = [round(x, 20) for x in list_logits]
        # with open(file_path, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(round_logits)
        #-------------------end----------------------
        return ret  

    # load init KV
    def get_global_hidden_and_mask(self, exc_length, local_h_k_len):
        '''
        函数会修改的变量:
        self._global_remainder_ed = self._global_remainder_st + exc_length
        self._global_remainder_st 如果要填充initKV则 += append_init_len
                                  否则不变
        self.init_k
        self.init_v
        self.global_buffer_init_st
        self.global_buffer_init_ed
        self.init_exc
        '''
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        global_remainder_ed = self._global_remainder_ed + exc_length #把本次要处理的那一块加进去
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # prepare init KV-Cache until it's full 每次从global_remainder KV 匀一些（超出n_local的部分）给 init KV 知道init KV填满
        if not self.init_exc and global_remainder_len == self.n_init: #global_remainder KV -> init KV
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = self.n_init - self.init_k.size(-2)
            
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            ) #初始化为0，一点一点往前加。
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True  # init KV-Cache is full

        self._global_remainder_ed = global_remainder_ed #无论如何，每次都增加exc_length
        self._global_remainder_st = global_remainder_st #st前边的是匀给init KV的，如果没有进入if则不变

        # load init KV -> global_h KV
        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:  # init KV haven't been loaded into global_h_kv
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)

        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        if self.init_exc and local_h_k_len > self.n_local:
            global_h_k = global_h_k[:, :, :init_ed, :]
            global_h_v = global_h_v[:, :, :init_ed, :]
        else:
            global_h_k = global_h_k[:, :, 0:0, :]
            global_h_v = global_h_v[:, :, 0:0, :]

        return global_h_k, global_h_v

    def _append(
        self,
        local_q, local_k, local_v, global_q,
    ):
        """calculate attention results 涉及到attention计算

        Args:
            local_q (_type_) : (batch_size, num_heads   , 196           or 13, dim_head)
            local_k (_type_) : (batch_size, num_heads_kv, 196 + n_local or 13, dim_head)
            local_v (_type_) : (batch_size, num_heads_kv, 196 + n_local or 13, dim_head)
            global_q (_type_): (batch_size, num_heads   , 196.          or 13, dim_head)

        Returns:
            chunk_o: (batch_size, num_heads, length, dim_head)
        """

        # apply RoPE to input QKV
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        # input Q attends to input + local KV (self.exc_block_size + self.n_init)
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            get_score=False, sliding_window=self.n_local
        )

        # load init KV
        local_h_k_len = local_h_k.size(-2)
        with torch.cuda.stream(GLOBAL_STREAM):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(exc_length=global_q.size(-2),local_h_k_len=local_h_k_len) #exc_length or 13

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # input Q attends to init KV
        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True,  # the final append operation
            get_score=False, 
            sliding_window=None,
            complement_sliding_window=True,
        )

        o, _ = attn.get_result()

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def _append_global(
        self
    ):
        """offload context memory,将global_remainder KV -> global_blocks、block_k
        函数会修改的变量:
        if not self.init_exc:函数无任何作用 self._global_remainder_ed不变
                                          self._global_remainder_st不变
        if self.init_exc: self._global_remainder_ed不变
                          self._global_remainder_st依次加self.block_size直到等于self._global_remainder_ed
                          self.global_blocks
                          self.block_k
                          self.num_global_block
        """

        global_remainder_ed = self._global_remainder_ed
        global_remainder_st = self._global_remainder_st

        global_remainder_len = global_remainder_ed - global_remainder_st

        # offload context KV to CPU
        if self.init_exc: #已经填充了init KV
            assert global_remainder_len % self.block_size == 0, f'global_remainder_len: {global_remainder_len}, block_size: {self.block_size}'
            while global_remainder_len > 0:
                global_remainder_len -= self.block_size
                
                # Context KV-Cache
                for u in range(self.num_units): #global_remainder -> global_blocks
                    #更新chunk2block
                    self.chunk2block[u][-1].append(len(self.global_blocks[u]))
                    self.global_blocks[u].append((
                        MemoryUnit(
                            (
                                self.global_remainder[0][u, :, global_remainder_st:global_remainder_st + self.block_size, :], 
                                self.global_remainder[1][u, :, global_remainder_st:global_remainder_st + self.block_size, :]
                            ),
                            self.cuda_cache, #所有MemoryUnit共用同一个cuda_cache
                            False,
                            self.pin_memory
                        )
                    )) #添加到global_blocks

                # NOTE: the average of global_remainder is used as the representative vector.
                global_block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :]
                global_block_k = self._from_group_kv(global_block_k)  # (batch_size, num_heads, length, dim_head)

                global_block_k = global_block_k.mean(dim=-2, keepdim=False)  # (batch_size, num_heads, dim_head)
                global_block_k = global_block_k.reshape(self.num_units, -1)  # (batch_size, num_heads * dim_head)
                global_block_k = global_block_k[:, None, :]  # (batch_size, 1, num_heads * dim_head)
                for u in range(self.num_units):
                    self.block_k[u].append(global_block_k[u])  #添加到block_k
                
                self.num_global_block += 1
                global_remainder_st += self.block_size #st每次往后移block_size

        self._global_remainder_ed = global_remainder_ed #ed并未改变
        self._global_remainder_st = global_remainder_st 

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
        compression_flag = False,
    ):  # encode video frames
        # local_q、global_q (batch, num_heads   , Nv*196 or 13, dim_head)
        # local_k、global_k (batch, num_heads_kv, Nv*196 or 13, dim_head)
        # local_v、global_v (batch, num_heads_kv, Nv*196 or 13, dim_head)
        # Pre-allocate GPU Memory.
        if not self.initialized:
            self.init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        input_length = local_q.size(-2) # Nv*196 or 13
        # print(f'input_length: {input_length/196}')
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # append local KV
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            ) # 先一次性加到global_remainder里，再按照exc_length依次处理

        # apply RoPE to global_q
        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        o_list = []
        init_flag = self.init_exc
        if self.init_exc:
            for u in range(self.num_units):
                self.chunk2block[u].append([])
        for st in range(0, input_length, self.exc_block_size):  # Process the input tokens in blocks. 每次处理exc_block_size
            '''
            self.init_exc = False: self._global_remainder_ed += exc_block_size
                                   self._global_remainder_st += append_init_len
            self.init_exc = True:  self._global_remainder_ed += exc_block_size
                                   self._global_remainder_st 依次加self.block_size 直到等于self._global_remainder_ed
            '''
            ed = min(st + self.exc_block_size, input_length)

            # calculate attention results
            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length
            chunk_o = self._append( #attends to input + local KV + init kv
                local_q[:, :, st:ed, :], #length:self.exc_block_size
                self.local_k[:, :, kv_st: kv_ed, :], #length: self.n_local + self.exc_block_size
                self.local_v[:, :, kv_st: kv_ed, :],
                global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            # offload context memory
            with torch.cuda.stream(GLOBAL_STREAM):
                self._append_global() #只有self.init_exc=True时才有意义

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)
        # 更新chunk_k
        if self.init_exc and init_flag:
            for u in range(self.num_units):
                chunk_st = self.chunk2block[u][-1][0]
                chunk_ed = self.chunk2block[u][-1][-1]
                chunk_k = self.block_k[u].get_mean(chunk_st, chunk_ed)
                self.chunk_k[u].append(chunk_k)
            self.num_global_chunk += 1
            #print(f'chunk_idx: {self.num_global_chunk-1},chunk_len:{len(self.chunk2block[0][-1])},chunk2block: {self.chunk2block[0][-1]}')
            # with open('/home/cyl476530/ReKV/encode.txt', 'a') as f:
            #     f.write(f'len:{len(self.chunk2block[0][-1])}\n')
        self.length += input_length
        #print(f'len_global_blocks: {len(self.global_blocks[0])}')
        # restrict the length of local KV-cache to self.n_local
        if not compression_flag:
            if self.local_k.size(-2) >= self.n_local:
                self.local_k = self.local_k[:, :, -self.n_local:, :]
                self.local_v = self.local_v[:, :, -self.n_local:, :]

        # update global remainder
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        assert not self.init_exc or self._global_remainder_st == self._global_remainder_ed, f'self.init_exc: {self.init_exc}, global_remainder_st: {self._global_remainder_st}, global_remainder_ed: {self._global_remainder_ed}'
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :],
                self.global_remainder[1][:, :, self._global_remainder_st:, :]
            )

        ret = torch.cat(o_list, dim=-2)

        return ret
    
    def size(self, *args, **kwargs):
        return self.length

    def calculate_cpu_memory(self):
        memory = 0
        for u in range(self.num_units):
            for block in self.global_blocks[u]:
                memory += block.calculate_cpu_memory()
        return memory
