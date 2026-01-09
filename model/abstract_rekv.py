import torch
from logzero import logger
import numpy as np
import json
import torch.nn.functional as F
import os
import time
'''
**************************************v1************************************
'''
def cossine_similarity1(self, a, b):
        a = a.squeeze(0) 
        b = b.squeeze(0)
        dot_product = torch.sum(a * b, dim=-1)
        norm_product = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
        cos_sim = dot_product / norm_product
        return cos_sim #(196, 196)

# def cossin_similarity2(a, b):
#     a = a.squeeze(0)
#     b = b.squeeze(0)
#     a = a.mean(dim=0,keepdim=False)
#     b = b.mean(dim=0,keepdim=False)
#     dot_product = torch.dot(a, b)
#     norm_product = torch.norm(a) * torch.norm(b)
#     return dot_product / norm_product
def cossin_similarity2(a, b):
    # a, b: (1, N, D) → we assume they are already on GPU
    # Compute mean token feature in one go
    a_mean = a.mean(dim=1).squeeze(0)  # (D,)
    b_mean = b.mean(dim=1).squeeze(0)  # (D,)
    dot = torch.dot(a_mean, b_mean)
    norm = torch.norm(a_mean) * torch.norm(b_mean)
    return dot / (norm + 1e-8)  # avoid div by zero

# def cossin_similarity3(a, b):
#     #(196,196)的对角线 3个一组
#     return dot_product

# def cossin_similarity4(a, b):
#     #(196,196)的对角线 超过某一个阈值的比例
#     return dot_product

def rescale_sum_match(arr, new_min=48, new_max=80):
    arr = np.array(arr, dtype=float)
    orig_sum = arr.sum()
    amin, amax = arr.min(), arr.max()
    n = len(arr)

    # 1. 全一样，直接返回
    if amin == amax:
        return arr.astype(int).tolist()

    # 2. 不全一样，线性缩放
    scaled = (arr - amin) / (amax - amin) * (new_max - new_min) + new_min
    scale_ratio = orig_sum / scaled.sum()
    mapped = scaled * scale_ratio
    mapped = np.round(mapped)
    mapped = np.clip(mapped, new_min, new_max)

    # 微调，确保sum完全一致
    diff = int(round(orig_sum - mapped.sum()))
    if diff != 0:
        # 按误差分布补偿
        indices = np.argsort(mapped) if diff > 0 else np.argsort(-mapped)
        for i in indices:
            # 在区间允许下加减
            if (diff > 0 and mapped[i] < new_max) or (diff < 0 and mapped[i] > new_min):
                mapped[i] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
                if diff == 0:
                    break
    return mapped.astype(np.int32).tolist()

def obtain_cdf_num(score_sum, target_num):
    all_sorted_scores = []
    for layer in range(len(score_sum)):
        score = score_sum[layer][0] #[num_token]

        mean = torch.mean(score)
        std = torch.std(score)
        normalized_score = (score - mean) / std
        score = F.softmax(normalized_score, dim=0)

        sorted_score, index = score.sort(descending=True)
        #sorted_score = sorted_score / sorted_score.sum() #这需要考虑一下
        sorted_score = sorted_score.cumsum(dim=0)
        all_sorted_scores.append(sorted_score) 
        
    all_sorted_scores = torch.stack(all_sorted_scores) #[num_layers, num_token]
    num_layers, num_token = all_sorted_scores.shape
    device = all_sorted_scores.device
    
    left = 0
    right = 1
    mid = 0
    while(right-left >  1e-6):
        mid = (left + right) / 2.0
        # 为每一层找到"首次累计信息超过mid"的下标
        idx = torch.searchsorted(all_sorted_scores, torch.full((num_layers,1), mid, device=device), right=False) #[num_layers, 1]
        count = idx.squeeze(-1) #每一层应该保留的token数  [num_layers]
        count[count < 1] = 1
    
        count = count.sum() #所有层应该保留的token总数
        if abs(count - target_num) < 5:
            break
        elif count > target_num:
            right = mid
        else:
            left = mid
    idx = torch.searchsorted(all_sorted_scores, torch.full((num_layers, 1), mid, device=device), right=False)
    count = idx.squeeze(-1)
    
    count[count < 1] = 1
    return count.cpu().numpy() 


# def merge(cur_chunk_feature,cur_chunk_cossim):
#     max_cossim = max(cur_chunk_cossim)
#     max_cossim_idx = cur_chunk_cossim.index(max_cossim)
#     pre_feature_idx = max_cossim_idx
#     post_feature_idx = max_cossim_idx+1
#     pre_feature = cur_chunk_feature[pre_feature_idx]
#     post_feature = cur_chunk_feature[post_feature_idx]
#     merged_feature = (pre_feature+post_feature)/2
#     #更新cur_chunk_cossim
#     if pre_feature_idx == 0:
#         cos_sim = cossin_similarity2(merged_feature,cur_chunk_feature[post_feature_idx+1])
#         cur_chunk_cossim.pop(0)
#         cur_chunk_cossim[0] = cos_sim
#     elif post_feature_idx == len(cur_chunk_feature)-1:
#         cos_sim = cossin_similarity2(merged_feature,cur_chunk_feature[pre_feature_idx-1])
#         cur_chunk_cossim.pop(-1)
#         cur_chunk_cossim[-1] = cos_sim
#     else:
#         cos_sim_pre = cossin_similarity2(merged_feature,cur_chunk_feature[pre_feature_idx-1])
#         cos_sim_post = cossin_similarity2(merged_feature,cur_chunk_feature[post_feature_idx+1])
#         cur_chunk_cossim[max_cossim_idx - 1] = cos_sim_pre
#         cur_chunk_cossim[max_cossim_idx + 1] = cos_sim_post
#         cur_chunk_cossim.pop(max_cossim_idx)
#     #更新cur_chunk_feature
#     cur_chunk_feature[pre_feature_idx] = merged_feature
#     cur_chunk_feature.pop(post_feature_idx)

#     return cur_chunk_feature,cur_chunk_cossim
def merge(cur_chunk_feature, cur_chunk_cossim):
    """
    Optimized merge using precomputed mean features to avoid redundant computation.
    Assumes cur_chunk_feature is a list of (1, N, D) tensors.
    """
    if len(cur_chunk_cossim) == 0:
        return cur_chunk_feature, cur_chunk_cossim

    # Find index of max cosine similarity
    max_cossim_idx = torch.argmax(torch.tensor(cur_chunk_cossim)).item()
    
    pre_idx = max_cossim_idx
    post_idx = max_cossim_idx + 1

    # Merge the two most similar frames (average)
    merged = (cur_chunk_feature[pre_idx] + cur_chunk_feature[post_idx]) / 2.0
    cur_chunk_feature[pre_idx] = merged
    del cur_chunk_feature[post_idx]  # faster than .pop() for last few elements

    # Update cosine similarities
    # Precompute mean features for all frames once
    means = [feat.mean(dim=1).squeeze(0) for feat in cur_chunk_feature]  # list of (D,)

    new_cossim = []
    for i in range(len(means) - 1):
        dot = torch.dot(means[i], means[i + 1])
        norm = torch.norm(means[i]) * torch.norm(means[i + 1])
        sim = dot / (norm + 1e-8)
        new_cossim.append(sim.item())

    return cur_chunk_feature, new_cossim

class Abstract_ReKV:
    processor = None
    kv_cache = None #kv_cache_manager

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, dataset, encode_prompt_ids, compression_ratio, compress_mode, compress_temp, retrieval_mode, retrieve_temp, max_chunk_size, min_chunk_size, segment_theta, segment_mode, chunk_global_sign, n_local, retrieve_size, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.dataset = dataset
        self.encode_prompt_ids = encode_prompt_ids
        self.compression_ratio = compression_ratio
        self.offline_compression_ratios = None
        self.compress_mode = compress_mode
        self.compress_temp = compress_temp
        self.retrieval_mode = retrieval_mode
        self.retrieve_temp = retrieve_temp
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.segment_theta = segment_theta
        self.segment_mode = segment_mode
        self.chunk_global_sign = chunk_global_sign
        self.n_local = n_local
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self): #编码初始prompt
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values #type ContextManager

    def _get_video_features(self, pixel_values_videos): #实现由子类提供 (B, F, C, H, W) -> (B, Nv*196, D)
        pass

    def _encode_video_chunk(self, video_features): #编码视频片段,只为更新kv_cache
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        #video_features: (B, Nv*196, D)
        if self.chunk_global_sign == "True":
            B = video_features.shape[0]
            D = video_features.shape[2]
            chunk_global_feature = video_features.view(B, -1, self.n_frame_tokens, D).mean(dim=1) #(B, 196, D)
            video_features = torch.cat([video_features, chunk_global_feature], dim=1) #(B, Nv*196+196, D)

        if self.compression_ratio < 1: #需要压缩
            if not isinstance(self.init_prompt_ids, torch.Tensor):
                self.encode_prompt_ids = torch.as_tensor([self.encode_prompt_ids], device=self.device)
            #print(f'self.mode:{self.mode}')
            if self.chunk_global_sign == "True":
                seq_len = video_features.shape[1] // self.n_frame_tokens - 1
            else:
                seq_len = video_features.shape[1] // self.n_frame_tokens
            if self.compress_mode == 'online': #在线压缩
                #encode + 得到每层的一个分数 compression_score [ 28 * [list] ]
                start_compress = time.perf_counter()
                output = self.language_model(input_ids=self.encode_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
                self.kv_cache = output.past_key_values
                compression_score = []
                for layer_kv in self.kv_cache:  # reset to default
                    compression_score.append(layer_kv.layer_compression_score)
                #根据上一步的分数计算每层的reserved_num以及compression_ratio
                target_num = int(len(self.kv_cache) * seq_len * (self.compression_ratio)) #总共需要保留的token数
                #print(f'num_layers: {len(self.kv_cache)}, seq_len: {seq_len}, target_num: {target_num}')
                reserved_nums = obtain_cdf_num(compression_score, target_num) #[num_layers]
                new_min =max(1, int((seq_len * self.compression_ratio) * (1 - self.compress_temp)))
                new_max =min(seq_len, int((seq_len * self.compression_ratio) * (1 + self.compress_temp)))
                reserved_nums = rescale_sum_match(reserved_nums, new_min, new_max)
                #print(f'reserved_nums: {reserved_nums}')
                reserved_nums = np.array(reserved_nums)
                reserved_nums[reserved_nums < 1] = 1
                assert(sum(reserved_nums) >= len(self.kv_cache))
                #reserved_nums = reserved_nums * target_num / reserved_nums.sum()
                #reserved_nums = reserved_nums.astype(np.int32)
                #reserved_nums[reserved_nums < 1] = 1
                #print(f'reserved_nums: {reserved_nums}')

                # jsonl_path = f'confs/{self.dataset}/compression/online/{str(self.compression_ratio)}-{str(self.compress_temp)}.jsonl'
                # os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                # with open(jsonl_path, 'a') as f:
                #     f.write(json.dumps([f / seq_len for f in reserved_nums]) + '\n')
                #     f.flush()
                #压缩KV
                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.reserved_num = reserved_nums[i]
                    layer_kv.compress_kv(video_features.shape[1])
                compress_time = time.perf_counter() - start_compress
                self.last_compress_time = compress_time
                logger.info(f"KV Compression Time: {compress_time * 1000:.2f} ms")

            elif self.compress_mode == 'offline': #离线模式
                #根据配置文件计算reserved_nums
                if self.offline_compression_ratios is None:
                    with open(f'confs/{self.dataset}/compression/offline/{str(self.compression_ratio)}.json', 'r') as f:
                        self.offline_compression_ratios = np.array(json.load(f))
                reserved_nums = (self.offline_compression_ratios * seq_len).round().astype(np.int32)
                reserved_nums[reserved_nums < 1] = 1
                #print(f'seq_len: {seq_len}, reserved_nums: {reserved_nums}')
                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.reserved_num = reserved_nums[i]
                # encode + 压缩
                output = self.language_model(input_ids=self.encode_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True) #对应patch.py里的model_forward
            else: # base
                reserved_num = np.round(seq_len * self.compression_ratio).astype(np.int32)
                if reserved_num < 1:
                    reserved_num = 1
                for layer_kv in self.kv_cache:
                    layer_kv.reserved_num = reserved_num
                output = self.language_model(input_ids=self.encode_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        else: #不需要压缩
            output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values #type ContextManager
    
    @torch.inference_mode()
    def encode_video(self, video, video_path=None):  # video: (Nv, H, W, 3)
        log_file_path = '/home/cyl476530/ReKV/segment.txt'
        # encode chunk by chunk，目标更新整个video的kv_cache
        num_frames = video.shape[0]
        if self.segment_mode == 'uniform':
            encode_chunk_size = self.max_chunk_size
            num_chunks = num_frames // encode_chunk_size

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * encode_chunk_size
                end_idx = start_idx + encode_chunk_size
                chunk_video = video[start_idx:end_idx]
                pixel_values_videos = self.processor.video_processor(chunk_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
                video_features = self._get_video_features(pixel_values_videos)
                self._encode_video_chunk(video_features) #每一次_encode_video_chunk都会init attention layer个数的contextmanager
                logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

            # Handle remaining frames
            remaining_frames = num_frames % encode_chunk_size
            if remaining_frames > 0:
                start_idx = num_chunks * encode_chunk_size
                end_idx = start_idx + remaining_frames
                remaining_video = video[start_idx:end_idx]
                pixel_values_videos = self.processor.video_processor(remaining_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
                remaining_features = self._get_video_features(pixel_values_videos)
                self._encode_video_chunk(remaining_features)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
        else: #semantic
            cur_chunk_feature = []
            cur_chunk_cossim = []
            cur_chunk_size = 0
            chunk_start_idx = 0

            total_overhead = 0.0
            feat_extract_overhead = 0.0  # time spent in frame-wise feature extraction (vs batched)
            logic_overhead = 0.0  

            with open(log_file_path, 'a') as f:
                #f.write('-----------------------start-------------------------\n')
                #f.write(f'video_path: {video_path}\n')
                for frame_idx in range(num_frames):
                    cur_frame = video[frame_idx:frame_idx+1]
                    start_feat = time.perf_counter()
                    pixel_values_frame = self.processor.video_processor(cur_frame, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
                    cur_frame_feature = self._get_video_features(pixel_values_frame)  #(1, 196, D)
                    
                    feat_time = time.perf_counter() - start_feat
                    feat_extract_overhead += feat_time

                    if not cur_chunk_feature:
                        cur_chunk_feature.append(cur_frame_feature)
                        past_frame_feature = cur_frame_feature
                        cur_chunk_size = 1
                        continue
                    start_logic = time.perf_counter()
                    cos_sim = cossin_similarity2(cur_frame_feature,past_frame_feature)
                    cur_chunk_cossim.append(cos_sim)
                    if cur_chunk_size < self.min_chunk_size or cos_sim > self.segment_theta:
                        cur_chunk_feature.append(cur_frame_feature)
                        if cur_chunk_size >= self.max_chunk_size:
                            cur_chunk_feature,cur_chunk_cossim  = merge(cur_chunk_feature,cur_chunk_cossim)
                        else:
                            cur_chunk_size += 1
                        past_frame_feature = cur_frame_feature
                    else:
                        f.write(f'{len(cur_chunk_feature)}\n')
                        cur_chunk_tensor = torch.cat(cur_chunk_feature, dim=1)
                        self._encode_video_chunk(cur_chunk_tensor)
                        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
                        #f.write(f'[{chunk_start_idx * 2}s,{(frame_idx - 1) * 2}s]')
                        #print(f"Chunk processed from frame {chunk_start_idx} to frame {frame_idx - 1}")
                        chunk_start_idx = frame_idx
                        cur_chunk_feature = [cur_frame_feature]
                        past_frame_feature = cur_frame_feature
                        cur_chunk_cossim = []
                        cur_chunk_size = 1
                    logic_time = time.perf_counter() - start_logic
                    logic_overhead += logic_time
                f.write(f'{len(cur_chunk_feature)}\n')
                cur_chunk_tensor = torch.cat(cur_chunk_feature, dim=1)
                self._encode_video_chunk(cur_chunk_tensor)
                logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
                #f.write('-----------------------end-------------------------\n')
                #f.write(f'[{chunk_start_idx * 2}s,{(frame_idx - 1) * 2}s]\n')
            total_overhead = feat_extract_overhead + logic_overhead
            logger.info(f"[Semantic Overhead] Total: {total_overhead:.3f}s | "
                    f"Frame-wise feat extract: {feat_extract_overhead:.3f}s | "
                    f"Logic (cos-sim/merge): {logic_overhead:.3f}s | "
                    f"Frames: {num_frames}")


    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128): #实现由子类提供
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory
