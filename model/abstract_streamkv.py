import torch
from logzero import logger
import numpy as np
import torch.nn.functional as F
import time

def cosine_similarity(a, b):
    a_mean = a.mean(dim=1).squeeze(0)
    b_mean = b.mean(dim=1).squeeze(0)
    dot = torch.dot(a_mean, b_mean)
    norm = torch.norm(a_mean) * torch.norm(b_mean)
    return dot / (norm + 1e-8)

def rescale_sum_match(arr, new_min=48, new_max=80):
    arr = np.array(arr, dtype=float)
    orig_sum = arr.sum()
    amin, amax = arr.min(), arr.max()
    n = len(arr)

    if amin == amax:
        return arr.astype(int).tolist()

    scaled = (arr - amin) / (amax - amin) * (new_max - new_min) + new_min
    scale_ratio = orig_sum / scaled.sum()
    mapped = scaled * scale_ratio
    mapped = np.round(mapped)
    mapped = np.clip(mapped, new_min, new_max)

    diff = int(round(orig_sum - mapped.sum()))
    if diff != 0:
        indices = np.argsort(mapped) if diff > 0 else np.argsort(-mapped)
        for i in indices:
            if (diff > 0 and mapped[i] < new_max) or (diff < 0 and mapped[i] > new_min):
                mapped[i] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
                if diff == 0:
                    break
    return mapped.astype(np.int32).tolist()

def obtain_cdf_num(score_sum, target_num):
    all_sorted_scores = []
    for layer in range(len(score_sum)):
        score = score_sum[layer][0]

        mean = torch.mean(score)
        std = torch.std(score)
        normalized_score = (score - mean) / std
        score = F.softmax(normalized_score, dim=0)

        sorted_score, index = score.sort(descending=True)
        sorted_score = sorted_score.cumsum(dim=0)
        all_sorted_scores.append(sorted_score)

    max_len = max(score.size(0) for score in all_sorted_scores)
    pad_value = torch.full((max_len,), float('inf'), device=all_sorted_scores[0].device, dtype=all_sorted_scores[0].dtype)
    sorted_scores = torch.stack(
        [torch.cat([score, pad_value[:max_len - score.size(0)]]) for score in all_sorted_scores],
        dim=0,
    )
    num_layers = sorted_scores.size(0)
    device = sorted_scores.device
    dtype = sorted_scores.dtype
    left = 0
    right = 1
    mid = 0

    while right-left > 1e-6:
        mid = (left + right)/2.0
        mid_tensor = torch.full((num_layers, 1), mid, dtype=dtype, device=device)
        pos = torch.searchsorted(sorted_scores, mid_tensor, right=False)
        pos = torch.clamp(pos, min=1)
        count = int(pos.sum().item())

        if abs(count - target_num) < 5:
            break
        elif count > target_num:
            right = mid
        else:
            left = mid

    mid_tensor = torch.full((num_layers, 1), mid, dtype=dtype, device=device)
    pos = torch.searchsorted(sorted_scores, mid_tensor, right=False)
    pos = torch.clamp(pos, min=1)
    idx_arr = pos.view(-1).cpu().tolist()
    return np.array(idx_arr)

def merge(cur_chunk_feature, cur_chunk_cossim):
    if len(cur_chunk_cossim) == 0:
        return cur_chunk_feature, cur_chunk_cossim

    max_cossim_idx = torch.argmax(torch.tensor(cur_chunk_cossim)).item()

    pre_idx = max_cossim_idx
    post_idx = max_cossim_idx + 1

    merged = (cur_chunk_feature[pre_idx] + cur_chunk_feature[post_idx]) / 2.0
    cur_chunk_feature[pre_idx] = merged
    del cur_chunk_feature[post_idx]

    new_cossim = list(cur_chunk_cossim)
    if pre_idx > 0:
        new_cossim[pre_idx - 1] = cosine_similarity(cur_chunk_feature[pre_idx - 1], merged).item()
    if post_idx < len(new_cossim):
        new_cossim[pre_idx] = cosine_similarity(merged, cur_chunk_feature[pre_idx + 1]).item()
        del new_cossim[post_idx]
    else:
        del new_cossim[pre_idx]

    return cur_chunk_feature, new_cossim

class Abstract_StreamKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, dataset, encode_prompt_ids, compression_ratio, compress_temp, retrieve_temp, max_segment_size, min_segment_size, segment_theta, segment_mode, use_segment_summary, n_local, retrieve_size, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.dataset = dataset
        self.encode_prompt_ids = encode_prompt_ids
        self.compression_ratio = compression_ratio
        self.compress_temp = compress_temp
        self.retrieve_temp = retrieve_temp
        self.max_segment_size = max_segment_size
        self.min_segment_size = min_segment_size
        self.segment_theta = segment_theta
        self.segment_mode = segment_mode
        self.use_segment_summary = use_segment_summary
        self.n_local = n_local
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos): 
        pass

    def _encode_video_chunk(self, video_features):
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        if self.use_segment_summary == "True":
            B = video_features.shape[0]
            D = video_features.shape[2]
            chunk_global_feature = video_features.view(B, -1, self.n_frame_tokens, D).mean(dim=1) 
            video_features = torch.cat([video_features, chunk_global_feature], dim=1)

        if self.compression_ratio < 1:
            if not isinstance(self.init_prompt_ids, torch.Tensor):
                self.encode_prompt_ids = torch.as_tensor([self.encode_prompt_ids], device=self.device)
            if self.use_segment_summary == "True":
                seq_len = video_features.shape[1] // self.n_frame_tokens - 1
            else:
                seq_len = video_features.shape[1] // self.n_frame_tokens
            start_compress = time.perf_counter()
            output = self.language_model(input_ids=self.encode_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
            self.kv_cache = output.past_key_values
            compression_score = []
            for layer_kv in self.kv_cache:
                compression_score.append(layer_kv.layer_compression_score)

            target_num = int(len(self.kv_cache) * seq_len * (self.compression_ratio)) 
            reserved_nums = obtain_cdf_num(compression_score, target_num) 
            new_min =max(1, int((seq_len * self.compression_ratio) * (1 - self.compress_temp)))
            new_max =min(seq_len, int((seq_len * self.compression_ratio) * (1 + self.compress_temp)))
            reserved_nums = rescale_sum_match(reserved_nums, new_min, new_max)
            reserved_nums = np.array(reserved_nums)
            reserved_nums[reserved_nums < 1] = 1
            assert(sum(reserved_nums) >= len(self.kv_cache))

            for i, layer_kv in enumerate(self.kv_cache):
                layer_kv.reserved_num = reserved_nums[i]
                layer_kv.compress_kv(video_features.shape[1])
            compress_time = time.perf_counter() - start_compress
            logger.info(f"KV Compression Time: {compress_time * 1000:.2f} ms")
        else:
            output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values 
    
    @torch.inference_mode()
    def encode_video(self, video, video_path=None):
        num_frames = video.shape[0]
        if self.segment_mode == 'uniform':
            encode_chunk_size = self.max_segment_size
            num_chunks = num_frames // encode_chunk_size

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * encode_chunk_size
                end_idx = start_idx + encode_chunk_size
                chunk_video = video[start_idx:end_idx]
                pixel_values_videos = self.processor.video_processor(chunk_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
                video_features = self._get_video_features(pixel_values_videos)
                self._encode_video_chunk(video_features)
                logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

            remaining_frames = num_frames % encode_chunk_size
            if remaining_frames > 0:
                start_idx = num_chunks * encode_chunk_size
                end_idx = start_idx + remaining_frames
                remaining_video = video[start_idx:end_idx]
                pixel_values_videos = self.processor.video_processor(remaining_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype) 
                remaining_features = self._get_video_features(pixel_values_videos)
                self._encode_video_chunk(remaining_features)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
        else:
            cur_chunk_feature = []
            cur_chunk_cossim = []
            cur_chunk_size = 0
            chunk_start_idx = 0

            for frame_idx in range(num_frames):
                cur_frame = video[frame_idx:frame_idx+1]
                pixel_values_frame = self.processor.video_processor(cur_frame, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
                cur_frame_feature = self._get_video_features(pixel_values_frame)

                if not cur_chunk_feature:
                    cur_chunk_feature.append(cur_frame_feature)
                    past_frame_feature = cur_frame_feature
                    cur_chunk_size = 1
                    continue
                cos_sim = cosine_similarity(cur_frame_feature,past_frame_feature)
                cur_chunk_cossim.append(cos_sim)
                if cur_chunk_size < self.min_segment_size or cos_sim > self.segment_theta:
                    cur_chunk_feature.append(cur_frame_feature)
                    if cur_chunk_size >= self.max_segment_size:
                        cur_chunk_feature,cur_chunk_cossim  = merge(cur_chunk_feature,cur_chunk_cossim)
                    else:
                        cur_chunk_size += 1
                    past_frame_feature = cur_frame_feature
                else:
                    cur_chunk_tensor = torch.cat(cur_chunk_feature, dim=1)
                    self._encode_video_chunk(cur_chunk_tensor)
                    logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
                    chunk_start_idx = frame_idx
                    cur_chunk_feature = [cur_frame_feature]
                    past_frame_feature = cur_frame_feature
                    cur_chunk_cossim = []
                    cur_chunk_size = 1
            cur_chunk_tensor = torch.cat(cur_chunk_feature, dim=1)
            self._encode_video_chunk(cur_chunk_tensor)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')


    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory
