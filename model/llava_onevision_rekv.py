import torch
import json
import numpy as np
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger
import torch.nn.functional as F
from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
import os
import time

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
    return mapped.astype(int).tolist()

def obtain_cdf_num(score_sum, target_num):
    # 每层处理自己的；最后拼到list/array即可
    all_sorted_scores = []
    for layer in range(len(score_sum)):
        score = score_sum[layer][0] #[num_token]

        mean = torch.mean(score)
        std = torch.std(score)
        normalized_score = (score - mean) / std
        score = F.softmax(normalized_score, dim=0)

        sorted_score, index = score.sort(descending=True)
        sorted_score = sorted_score.cumsum(dim=0)
        all_sorted_scores.append(sorted_score)
        
    num_layers = len(all_sorted_scores)
    device = all_sorted_scores[0].device
    left = 0
    right = 1
    mid = 0

    # 二分法搜索全局mid
    while right-left > 1e-6:
        mid = (left + right)/2.0
        # 每一层分别找首次累计信息超过mid的下标
        idx = []
        for x in all_sorted_scores:
            # torch.searchsorted要求一样的dtype
            mid_tensor = torch.tensor([mid], dtype=x.dtype, device=x.device)
            pos = torch.searchsorted(x, mid_tensor, right=False).item()
            idx.append(max(1, pos)) #最少保留1

        count = sum(idx)
        if abs(count - target_num) < 5:
            break
        elif count > target_num:
            right = mid
        else:
            left = mid

    idx_arr = []
    for x in all_sorted_scores:
        mid_tensor = torch.tensor([mid], dtype=x.dtype, device=x.device)
        pos = torch.searchsorted(x, mid_tensor, right=False).item()
        idx_arr.append(max(1, pos))
    # 你要的 numpy 数组
    return np.array(idx_arr)

class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, dataset, encode_prompt_ids, compression_ratio, compress_mode, compress_temp, retrieval_mode, retrieve_temp, max_chunk_size, min_chunk_size, segment_theta, segment_mode, chunk_global_sign, n_local, retrieve_size, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, dataset, encode_prompt_ids, compression_ratio, compress_mode, compress_temp, retrieval_mode, retrieve_temp, max_chunk_size, min_chunk_size, segment_theta, segment_mode, chunk_global_sign, n_local, retrieve_size, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos): #视觉特征提取
        batch_size, frames, channels, height, width = pixel_values_videos.shape # 输入为(B, F, C, H, W)
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width) #(B*F, C, H, W)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True) #(B*F, 196, 768)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer] #(B*F, 196, 768)

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None, streamer=None):
        ttft = None
        decoding_throughput = None

        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids #先用question来retrieval
        input_ids = torch.as_tensor([input_ids], device=device)

        for layer_kv in self.kv_cache:  # activate retrieval mode
            layer_kv.set_retrieval()

        if retrieved_indices is None:  # Internal retrieval 基本都是None
            if self.retrieval_mode == 'online':
                for layer_kv in self.kv_cache:
                    layer_kv.set_retrieval_prefill()
                #先每层topk计算分数
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
                retrieval_score = []
                for layer_kv in self.kv_cache:
                    layer_kv.reset_retrieval_prefill()
                    #print(layer_kv.retrieval_prefill)
                    retrieval_score.append(layer_kv.layer_retrieval_score)
                #再用分数计算retrieval_nums

                start_retrieval_alloc = time.perf_counter()

                target_num = len(self.kv_cache) * self.retrieve_size
                #print(f'retrieval_score: {retrieval_score}')
                retrieval_nums = obtain_cdf_num(retrieval_score, target_num)
                #print(f'retrieval_nums: {retrieval_nums}')
                retrieval_nums = rescale_sum_match(retrieval_nums,max(self.retrieve_size*(1-self.retrieve_temp),1),min(self.retrieve_size*(1+self.retrieve_temp),128))
                # jsonl_path = f'confs/{self.dataset}/retrieval/online/{str(self.retrieve_temp)}.jsonl'
                # os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                # with open(jsonl_path, 'a') as f:
                #     f.write(json.dumps(retrieval_nums) + '\n')
                #     f.flush()
                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.retrieval_num = retrieval_nums[i]
                retrieval_alloc_time = time.perf_counter() - start_retrieval_alloc
                self.last_retrieval_alloc_time = retrieval_alloc_time
                logger.info(f"[Retrieval Budget Allocation] Time: {retrieval_alloc_time * 1000:.2f} ms, "
                         f"Layers: {len(self.kv_cache)}, Target tokens: {target_num}")
                #最后再进行真正的retrieval
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            elif self.retrieval_mode == 'offline':
                with open(f'confs/{self.dataset}/retrieval/offline.json', 'r') as f:
                    retrieval_nums = np.array(json.load(f))
                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.retrieval_num = retrieval_nums[i]
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            else: #base
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh) type:tuple
        else:  # External retrieval
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        output_ids = []
        stopped = False
        
        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids #用prompt来进行prefill
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]

            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]
            
            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output


def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_encode=None, n_local=None, dataset=None, compression_ratio=None, compress_mode=None, compress_temp=None, encode_mode=None, retrieval_mode=None, retrieve_temp=None, retrieve_local=None, retrieve_local_size=None, max_chunk_size=None, min_chunk_size=None, segment_theta=None, chunk_global_sign=None, segment_mode=None, retrieve_size=64, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant. Please actively track and remember current objects, actions, on-screen texts/numbers, temporal changes, and any anomalies. Focus on capturing key visual clues, filtering misleading or outdated context, and keep in mind prior key events to support sequential and time-critical queries or proactive outputs throughout the streaming video.<|im_end|><|im_start|>user '
    encode_prompt = 'For this event segment in the video, please prioritize retaining the following visual information:1)Key objects, people, and their attributes, and any changes or transitions during the event.2)Major actions, state transitions, and interactions (e.g., who/what is doing what, when, and where), including the precise timing of key moments.3)On-screen text, numbers, and all visible cues (such as labels, scores, timer, signs, etc.).4)Causal relationships, outcome(s), anomalies, and salient shifts within the segment (e.g., cause-effect chains, rare or surprising events, turning points).5)Counts and spatial arrangements of important objects or people, as well as layout details that may support summarization, prediction, or sequential reasoning in later queries.6)Any misleading, ambiguous, or easily confused visual clues—ensure enough context is preserved to distinguish the current event state from similar past/future states.'
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    encode_prompt_ids = processor.tokenizer(encode_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_encode': encode_prompt_ids.shape[1] if n_encode is None else n_encode, # 编码初始prompt的长度
        'chunk_global_sign': chunk_global_sign, 
        'n_local': n_local,
        'compression_ratio': compression_ratio,
        'compress_mode': compress_mode,
        'encode_mode': encode_mode, 
        'retrieval_mode': retrieval_mode,
        'retrieve_local': retrieve_local,
        'retrieve_local_size': retrieve_local_size, 
        'fattn': True,
        'block_size': n_frame_tokens, # 单帧token数
        'retrieve_size': retrieve_size,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens, #单帧token数
        'pin_memory': True,
    }
    model = LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        dataset=dataset,
        encode_prompt_ids=encode_prompt_ids,
        compression_ratio=compression_ratio,
        compress_mode=compress_mode,
        compress_temp=compress_temp,
        retrieval_mode=retrieval_mode,
        retrieve_temp=retrieve_temp,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        segment_theta=segment_theta,
        segment_mode=segment_mode,
        chunk_global_sign=chunk_global_sign,
        n_local=n_local,
        retrieve_size=retrieve_size,
        chunk_size=chunk_size,
    )
    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor