import torch
from logzero import logger
import numpy as np
import json
import torch.nn.functional as F
import os

def cosine_sim(a, b):
    a = a.squeeze(0)
    b = b.squeeze(0)
    a = a.mean(dim=0,keepdim=False)
    b = b.mean(dim=0,keepdim=False)
    dot_product = torch.dot(a, b)
    norm_product = torch.norm(a) * torch.norm(b)
    return dot_product / norm_product

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

def layer_adaptive_allocation(score_sum, target_num):
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
        
    all_sorted_scores = torch.stack(all_sorted_scores)
    num_layers, num_token = all_sorted_scores.shape
    device = all_sorted_scores.device
    
    left = 0
    right = 1
    mid = 0
    while(right-left >  1e-6):
        mid = (left + right) / 2.0
        idx = torch.searchsorted(all_sorted_scores, torch.full((num_layers,1), mid, device=device), right=False)
        count = idx.squeeze(-1)
        count[count < 1] = 1
    
        count = count.sum()
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


def segment_merge(cur_segment_feature,cur_segment_cossim):
    max_cossim = max(cur_segment_cossim)
    max_cossim_idx = cur_segment_cossim.index(max_cossim)
    pre_feature_idx = max_cossim_idx
    post_feature_idx = max_cossim_idx+1
    pre_feature = cur_segment_feature[pre_feature_idx]
    post_feature = cur_segment_feature[post_feature_idx]
    merged_feature = (pre_feature+post_feature)/2
    if pre_feature_idx == 0:
        cos_sim = cosine_sim(merged_feature,cur_segment_feature[post_feature_idx+1])
        cur_segment_cossim.pop(0)
        cur_segment_cossim[0] = cos_sim
    elif post_feature_idx == len(cur_segment_feature)-1:
        cos_sim = cosine_sim(merged_feature,cur_segment_feature[pre_feature_idx-1])
        cur_segment_cossim.pop(-1)
        cur_segment_cossim[-1] = cos_sim
    else:
        cos_sim_pre = cosine_sim(merged_feature,cur_segment_feature[pre_feature_idx-1])
        cos_sim_post = cosine_sim(merged_feature,cur_segment_feature[post_feature_idx+1])
        cur_segment_cossim[max_cossim_idx - 1] = cos_sim_pre
        cur_segment_cossim[max_cossim_idx + 1] = cos_sim_post
        cur_segment_cossim.pop(max_cossim_idx)
    cur_segment_feature[pre_feature_idx] = merged_feature
    cur_segment_feature.pop(post_feature_idx)

    return cur_segment_feature,cur_segment_cossim

class Abstract_StreamKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, dataset, guidance_prompt_ids, compression_ratio, compress_mode, compress_temp, retrieve_mode, retrieve_temp, max_segment_size, min_segment_size, segment_theta, segment_mode, segment_summary_sign, n_local, retrieve_size, segment_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.dataset = dataset
        self.guidance_prompt_ids = guidance_prompt_ids
        self.compression_ratio = compression_ratio
        self.offline_compression_ratios = None
        self.compress_mode = compress_mode
        self.compress_temp = compress_temp
        self.retrieve_mode = retrieve_mode
        self.retrieve_temp = retrieve_temp
        self.max_segment_size = max_segment_size
        self.min_segment_size = min_segment_size
        self.segment_theta = segment_theta
        self.segment_mode = segment_mode
        self.segment_summary_sign = segment_summary_sign
        self.n_local = n_local
        self.retrieve_size = retrieve_size
        self.segment_size = segment_size

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

    def _encode_video_segment(self, video_features): 
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        if self.segment_summary_sign == "True":
            B = video_features.shape[0]
            D = video_features.shape[2]
            segment_global_feature = video_features.view(B, -1, self.n_frame_tokens, D).mean(dim=1)
            video_features = torch.cat([video_features, segment_global_feature], dim=1)

        if self.compression_ratio < 1:
            if not isinstance(self.init_prompt_ids, torch.Tensor):
                self.guidance_prompt_ids = torch.as_tensor([self.guidance_prompt_ids], device=self.device)
            if self.segment_summary_sign == "True":
                seq_len = video_features.shape[1] // self.n_frame_tokens - 1
            else:
                seq_len = video_features.shape[1] // self.n_frame_tokens
            if self.compress_mode == 'online': 
                output = self.language_model(input_ids=self.guidance_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
                self.kv_cache = output.past_key_values
                compression_score = []
                for layer_kv in self.kv_cache:
                    compression_score.append(layer_kv.layer_compression_score)
                target_num = int(len(self.kv_cache) * seq_len * (self.compression_ratio)) 
                reserved_nums = layer_adaptive_allocation(compression_score, target_num) 
                new_min =max(1, int((seq_len * self.compression_ratio) * (1 - self.compress_temp)))
                new_max =min(seq_len, int((seq_len * self.compression_ratio) * (1 + self.compress_temp)))
                reserved_nums = rescale_sum_match(reserved_nums, new_min, new_max)
                reserved_nums[reserved_nums < 1] = 1
                assert(sum(reserved_nums) >= len(self.kv_cache))

                jsonl_path = f'jsonl'
                os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                with open(jsonl_path, 'a') as f:
                    f.write(json.dumps([f / seq_len for f in reserved_nums]) + '\n')
                    f.flush()

                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.reserved_num = reserved_nums[i]
                    layer_kv.compress_kv(video_features.shape[1])

            elif self.compress_mode == 'offline':
                if self.offline_compression_ratios is None:
                    with open(f'json', 'r') as f:
                        self.offline_compression_ratios = np.array(json.load(f))
                reserved_nums = (self.offline_compression_ratios * seq_len).round().astype(np.int32)
                reserved_nums[reserved_nums < 1] = 1
                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.reserved_num = reserved_nums[i]
                output = self.language_model(input_ids=self.guidance_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
            else:
                reserved_num = np.round(seq_len * self.compression_ratio).astype(np.int32)
                if reserved_num < 1:
                    reserved_num = 1
                for layer_kv in self.kv_cache:
                    layer_kv.reserved_num = reserved_num
                output = self.language_model(input_ids=self.guidance_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        else: 
            output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values 
    
    @torch.inference_mode()
    def encode_video(self, video):
        num_frames = video.shape[0]
        if self.segment_mode == 'uniform':
            encode_segment_size = self.max_segment_size
            num_segments = num_frames // encode_segment_size

            for segment_idx in range(num_segments):
                start_idx = segment_idx * encode_segment_size
                end_idx = start_idx + encode_segment_size
                segment_video = video[start_idx:end_idx]
                pixel_values_videos = self.processor.video_processor(segment_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype) 
                video_features = self._get_video_features(pixel_values_videos)
                self._encode_video_segment(video_features)

            remaining_frames = num_frames % encode_segment_size
            if remaining_frames > 0:
                start_idx = num_segments * encode_segment_size
                end_idx = start_idx + remaining_frames
                remaining_video = video[start_idx:end_idx]
                pixel_values_videos = self.processor.video_processor(remaining_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype) 
                remaining_features = self._get_video_features(pixel_values_videos)
                self._encode_video_segment(remaining_features)
        else: 
            cur_segment_feature = []
            cur_segment_cossim = []
            cur_segment_size = 0
            segment_start_idx = 0
            for frame_idx in range(num_frames):
                cur_frame = video[frame_idx:frame_idx+1]
                pixel_values_frame = self.processor.video_processor(cur_frame, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
                cur_frame_feature = self._get_video_features(pixel_values_frame)  
                if not cur_segment_feature:
                    cur_segment_feature.append(cur_frame_feature)
                    past_frame_feature = cur_frame_feature
                    cur_segment_size = 1
                    continue
                cos_sim = cosine_sim(cur_frame_feature,past_frame_feature)
                cur_segment_cossim.append(cos_sim)
                if cur_segment_size < self.min_segment_size or cos_sim > self.segment_theta:
                    cur_segment_feature.append(cur_frame_feature)
                    if cur_segment_size >= self.max_segment_size:
                        cur_segment_feature,cur_segment_cossim  = segment_merge(cur_segment_feature,cur_segment_cossim)
                    else:
                        cur_segment_size += 1
                    past_frame_feature = cur_frame_feature
                else:
                    cur_segment_tensor = torch.cat(cur_segment_feature, dim=1)
                    self._encode_video_segment(cur_segment_tensor)
                    segment_start_idx = frame_idx
                    cur_segment_feature = [cur_frame_feature]
                    past_frame_feature = cur_frame_feature
                    cur_segment_cossim = []
                    cur_segment_size = 1
            cur_segment_tensor = torch.cat(cur_segment_feature, dim=1)
            self._encode_video_segment(cur_segment_tensor)


    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass
