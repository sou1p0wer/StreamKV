import torch
from typing import Optional, Tuple
from .dot_production_attention import get_multi_stage_dot_production_attention

class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units
        self.unit_size = unit_size
        self.dtype = dtype
        self.data = torch.empty(
            (num_units, unit_size),
            device = "cuda",
            dtype=dtype
        )
        self.idle_set = set(list(range(num_units)))

    def alloc(self):
        assert len(self.idle_set) > 0
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)

class MemoryUnit:
    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        if pin_memory:
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)
        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
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

    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def calculate_cpu_memory(self):
        return len(self.cpu_data) * self.cpu_data[0].numel() * self.cpu_data[0].element_size()

class VectorTensor:
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

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size, f'{tensor.size(1)}, {self.hidden_size}'
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

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

    def get_data(self):
        return self.data[:self.length, ...]

    def get_window_cosine_similarity(self, window_size, tensor, use_segment_summary):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        if use_segment_summary == "True":
            key = self.data[self.length-window_size:self.length-1].float()
        else:
            key = self.data[self.length-window_size:self.length].float()
        query = tensor[None, :].float()
        logits = torch.matmul(query, key.T)[0]
        assert logits.dim() == 1 and logits.size(0) == window_size-1 if use_segment_summary=="True" else window_size ,f'{logits.dim()}, {logits.size(0)},{window_size}'
        return logits

    def get_cosine_similarity(self, tensor: torch.Tensor):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        key = self.data[:self.length].float()
        query = tensor[None, :].float()

        logits = torch.matmul(query, key.T)[0]

        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def __len__(self):
        return self.length


GLOBAL_STREAM = None


class ContextManager:
    def __init__(self, 
                 position_embedding,
                 n_init, n_local, use_segment_summary,
                 block_size, max_cached_block, retrieve_size, chunk_size, exc_block_size,
                 retrieve_local, retrieve_local_size,
                 fattn: bool = False,
                 async_global_stream: bool = False,
                 pin_memory: bool = False,
    ):

        self.length = 0
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.use_segment_summary = use_segment_summary
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.max_retrieve_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local
        self.retrieve_size = retrieve_size
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        self.retrieve_local = retrieve_local
        self.retrieve_local_size = retrieve_local_size
        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.reset_retrieval()

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

    def _from_group_kv(self, tensor):
        assert tensor.dim() == 4 
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_units, self.unit_size_kv, 1, length, dim_head))
        tensor = tensor.expand((self.num_units, self.unit_size_kv, num_group, length, dim_head)).reshape((self.num_units, self.num_heads, length, dim_head))
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

        self.global_blocks = [[] for _ in range(self.num_units)]
        self.cached_blocks = [{} for _ in range(self.num_units)]
        self.num_global_block = 0

        self.block_k = [VectorTensor(
            dim_head * self.unit_size, global_k.dtype, global_k.device
        ) for _ in range(self.num_units)]

        self.local_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)
        self.local_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        self.global_remainder = (
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )

        self.init_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        buffer_len = self.max_retrieve_block * self.block_size + self.n_init
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
        )

        self.chunk2block = [[] for _ in range(self.num_units)]
        self.layer_compression_score = None
        self.layer_retrieval_score = None
        self.reserved_num = None
        self.retrieval_num = None

        self.initialized = True
    def set_retrieval_prefill(self):
        self.retrieval_prefill = True

    def reset_retrieval_prefill(self):
        self.retrieval_prefill = False

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.retrieved_block_indices = None
        self.to_retrieve = False

    def set_retrieved_block_indices(self, retrieved_block_indices):
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices
    
    def calculate_score(self, video_len, prompt):
        num_frames = video_len // self.block_size
        prompt = prompt.mean(dim=2, keepdim=False)
        assert prompt.shape == (self.num_units, self.unit_size, self.dim_head)
        prompt = prompt.reshape(self.num_units, self.dim_head * self.unit_size)
        logits = torch.stack([self.block_k[u].get_window_cosine_similarity(num_frames,prompt[u],self.use_segment_summary) for u in range(self.num_units)])
        self.layer_compression_score = logits

    def compress_kv(self, video_len):
        num_frames = video_len // self.block_size

        logits = self.layer_compression_score
        assert logits.shape == (self.num_units, num_frames - 1) if self.use_segment_summary == "True" else (self.num_units, num_frames)
        reserved_num = self.reserved_num
        
        num_frames_after_compressed = reserved_num
        all_indices = torch.arange(num_frames).to(logits.device)
        kept_indices = logits.topk(num_frames_after_compressed, dim=1).indices
        if self.use_segment_summary == "True":
            global_indices = torch.full((self.num_units, 1), num_frames-1, dtype=kept_indices.dtype, device=kept_indices.device)
            kept_indices = torch.cat([kept_indices, global_indices], dim=1)
        kept_indices = kept_indices.sort(dim=1)[0]
        evicted_indices = [all_indices[~torch.isin(all_indices, kept_indices[u])] for u in range(self.num_units)]
        evicted_indices = [ekv.sort(descending=True)[0] for ekv in evicted_indices]
        offset = self.num_global_block - num_frames
        evicted_indices = [ekv + offset for ekv in evicted_indices]
        for u in range(self.num_units):
            for idx in evicted_indices[u]:
                if idx in self.cached_blocks[u].keys():
                    self.global_blocks[u][idx].offload()
                    self.cached_blocks[u].pop(idx)
                del self.global_blocks[u][idx]
                self.num_global_block -= 1
            chunk_st = self.chunk2block[u][-1][0]
            self.chunk2block[u][-1] = list(range(chunk_st, chunk_st + num_frames_after_compressed))
            self.block_k[u].delete(evicted_indices[u])


    def get_retrieved_kv(self,query=None):
        """retrieve context blocks with retrieved_block_indices
        query: (batch_size, num_heads, seq_len, dim_head)
        return [init_k, retrieved_k] and the respective v
        """

        if query is not None:
            block_topk = self._calc_block_topk(query)
            self.set_retrieved_block_indices(block_topk)

        assert len(self.retrieved_block_indices) == self.num_units

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        with torch.cuda.stream(GLOBAL_STREAM):
            for u in range(self.num_units):
                num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                for b_idx in self.retrieved_block_indices[u]:
                    if b_idx not in self.cached_blocks[u]:
                        num_remove += 1
                self._remove_lru_blocks(u, num_remove, self.retrieved_block_indices[u])

            self.load_count += 1
            for u in range(self.num_units):
                for b_idx in self.retrieved_block_indices[u]:
                    self.cached_blocks[u][b_idx] = self.load_count
            
            init_st = 0
            init_ed = init_st + self.init_k.size(-2)
            ed = init_ed
            assert self.global_buffer_init_st == init_st or self.global_buffer_init_ed == init_ed

            for u in range(self.num_units):
                assert self.retrieved_block_indices[u][-1] < self.num_global_block, f'{self.retrieved_block_indices[u][-1]}, {self.num_global_block}'
                for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                    st = init_ed + cnt * self.block_size
                    ed = st + self.block_size
                    self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))

            global_h_k = global_h_k[:, :, :ed, :]
            global_h_v = global_h_v[:, :, :ed, :]

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        assert global_h_k.size(-2) <= self.n_init + self.max_retrieve_block * self.block_size
        return global_h_k, global_h_v

    def _calc_block_topk(
        self,global_h_q
    ):
        global_h_q = global_h_q.mean(dim=2, keepdim=False)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(self.num_units, self.dim_head * self.unit_size)
        logits = None
        if self.retrieval_prefill:
            retrieval_num = self.retrieve_size
        else:
            retrieval_num = self.retrieval_num
        if self.retrieve_local == 'False':
            logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
            if self.retrieval_prefill:
                self.layer_retrieval_score = logits

            if self.num_global_block <= retrieval_num:
                ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
            else:
                ret = logits.topk(retrieval_num, dim=1).indices
                ret = ret.sort(dim=1)[0]
                ret = ret.cpu().tolist()
        else:
            logits = torch.stack([self.block_k[u].get_cosine_similarity(global_h_q[u]) for u in range(self.num_units)])
            if self.retrieval_prefill:
                self.layer_retrieval_score = logits

            ret = []
            for u in range(self.num_units):
                if self.num_global_block <= retrieval_num:
                    ret.append(list(range(len(self.global_blocks[u]))))
                else:
                    if retrieval_num >= self.retrieve_local_size:
                        new_retrieval_num = retrieval_num - self.retrieve_local_size
                        retrieve_local = list(range(self.num_global_block - self.retrieve_local_size, self.num_global_block))
                        new_logits = logits[u][:-self.retrieve_local_size]
                        top_idx = new_logits.topk(new_retrieval_num).indices.cpu().tolist()
                        indices = sorted(top_idx + retrieve_local)
                    else:
                        indices = list(range(self.num_global_block - retrieval_num, self.num_global_block))
                    ret.append(indices)
        return ret

    def get_global_hidden_and_mask(self, exc_length, local_h_k_len):
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        if not self.init_exc and global_remainder_len == self.n_init:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = self.n_init - self.init_k.size(-2)
            
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:
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
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            get_score=False, sliding_window=self.n_local
        )

        local_h_k_len = local_h_k.size(-2)
        with torch.cuda.stream(GLOBAL_STREAM):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(exc_length=global_q.size(-2),local_h_k_len=local_h_k_len)

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True,
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
        global_remainder_ed = self._global_remainder_ed
        global_remainder_st = self._global_remainder_st

        global_remainder_len = global_remainder_ed - global_remainder_st

        if self.init_exc: 
            assert global_remainder_len % self.block_size == 0, f'global_remainder_len: {global_remainder_len}, block_size: {self.block_size}'
            while global_remainder_len > 0:
                global_remainder_len -= self.block_size

                for u in range(self.num_units):
                    self.chunk2block[u][-1].append(len(self.global_blocks[u]))
                    self.global_blocks[u].append((
                        MemoryUnit(
                            (
                                self.global_remainder[0][u, :, global_remainder_st:global_remainder_st + self.block_size, :],
                                self.global_remainder[1][u, :, global_remainder_st:global_remainder_st + self.block_size, :]
                            ),
                            self.cuda_cache,
                            False,
                            self.pin_memory
                        )
                    ))

                global_block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :]
                global_block_k = self._from_group_kv(global_block_k)

                global_block_k = global_block_k.mean(dim=-2, keepdim=False)
                global_block_k = global_block_k.reshape(self.num_units, -1)
                global_block_k = global_block_k[:, None, :]
                for u in range(self.num_units):
                    self.block_k[u].append(global_block_k[u])
                
                self.num_global_block += 1
                global_remainder_st += self.block_size

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st 

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        if not self.initialized:
            self.init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        input_length = local_q.size(-2)
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        o_list = []
        if self.init_exc:
            for u in range(self.num_units):
                self.chunk2block[u].append([])
        for st in range(0, input_length, self.exc_block_size):
            ed = min(st + self.exc_block_size, input_length)

            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length
            chunk_o = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st: kv_ed, :],
                self.local_v[:, :, kv_st: kv_ed, :],
                global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            with torch.cuda.stream(GLOBAL_STREAM):
                self._append_global()

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)
        self.length += input_length
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :]

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
