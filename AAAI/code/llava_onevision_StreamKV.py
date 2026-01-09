import torch
import json
import numpy as np
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger
import torch.nn.functional as F
from model.patch import patch_hf
from model.abstract_StreamKV import Abstract_StreamKV
import os

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
    return mapped.astype(int).tolist()

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
        
    num_layers = len(all_sorted_scores)
    device = all_sorted_scores[0].device
    left = 0
    right = 1
    mid = 0

    while right-left > 1e-6:
        mid = (left + right)/2.0
        idx = []
        for x in all_sorted_scores:
            mid_tensor = torch.tensor([mid], dtype=x.dtype, device=x.device)
            pos = torch.searchsorted(x, mid_tensor, right=False).item()
            idx.append(max(1, pos)) 

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
    return np.array(idx_arr)

class LlavaOneVision_StreamKV(LlavaOnevisionForConditionalGeneration, Abstract_StreamKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, dataset, guidance_prompt_ids, compression_ratio, compress_mode, compress_temp, retrieve_mode, retrieve_temp, max_chunk_size, min_chunk_size, segment_theta, segment_mode, segment_summary_sign, n_local, retrieve_size, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_StreamKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, dataset, guidance_prompt_ids, compression_ratio, compress_mode, compress_temp, retrieve_mode, retrieve_temp, max_chunk_size, min_chunk_size, segment_theta, segment_mode, segment_summary_sign, n_local, retrieve_size, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos): 
        batch_size, frames, channels, height, width = pixel_values_videos.shape 
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width) 
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  
        return video_features

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        for layer_kv in self.kv_cache: 
            layer_kv.set_retrieval()

        if retrieved_indices is None: 
            if self.retrieve_mode == 'online':
                for layer_kv in self.kv_cache:
                    layer_kv.set_retrieval_prefill()
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
                retrieval_score = []
                for layer_kv in self.kv_cache:
                    layer_kv.reset_retrieval_prefill()
                    retrieval_score.append(layer_kv.layer_retrieval_score)
                target_num = len(self.kv_cache) * self.retrieve_size
                retrieval_nums = layer_adaptive_allocation(retrieval_score, target_num)
                retrieval_nums = rescale_sum_match(retrieval_nums,max(self.retrieve_size*(1-self.retrieve_temp),1),min(self.retrieve_size*(1+self.retrieve_temp),128))
                jsonl_path = f'online.jsonl'
                os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                with open(jsonl_path, 'a') as f:
                    f.write(json.dumps(retrieval_nums) + '\n')
                    f.flush()
                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.retrieval_num = retrieval_nums[i]
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            elif self.retrieve_mode == 'offline':
                with open(f'offline.json', 'r') as f:
                    retrieval_nums = np.array(json.load(f))
                for i, layer_kv in enumerate(self.kv_cache):
                    layer_kv.retrieval_num = retrieval_nums[i]
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            else:
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values 
        else: 
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values 

        for layer_kv in self.kv_cache: 
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0: 
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids 
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else: 
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


def load_model(model_path='llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_guidance=None, n_local=None, dataset=None, compression_ratio=None, compress_mode=None, compress_temp=None, encode_mode=None, retrieve_mode=None, retrieve_temp=None, retrieve_local=None, retrieve_local_size=None, max_chunk_size=None, min_chunk_size=None, segment_theta=None, segment_summary_sign=None, segment_mode=None, retrieve_size=64, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant. Please actively track and remember current objects, actions, on-screen texts/numbers, temporal changes, and any anomalies. Focus on capturing key visual clues, filtering misleading or outdated context, and keep in mind prior key events to support sequential and time-critical queries or proactive outputs throughout the streaming video.<|im_end|><|im_start|>user '
    guidance_prompt = 'For this event segment in the video, please prioritize retaining the following visual information:1)Key objects, people, and their attributes, and any changes or transitions during the event.2)Major actions, state transitions, and interactions (e.g., who/what is doing what, when, and where), including the precise timing of key moments.3)On-screen text, numbers, and all visible cues (such as labels, scores, timer, signs, etc.).4)Causal relationships, outcome(s), anomalies, and salient shifts within the segment (e.g., cause-effect chains, rare or surprising events, turning points).5)Counts and spatial arrangements of important objects or people, as well as layout details that may support summarization, prediction, or sequential reasoning in later queries.6)Any misleading, ambiguous, or easily confused visual clues—ensure enough context is preserved to distinguish the current event state from similar past/future states.'
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    guidance_prompt_ids = processor.tokenizer(guidance_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_guidance': guidance_prompt_ids.shape[1] if n_guidance is None else n_guidance,
        'segment_summary_sign': segment_summary_sign, 
        'n_local': n_local,
        'compression_ratio': compression_ratio,
        'compress_mode': compress_mode,
        'encode_mode': encode_mode, 
        'retrieve_mode': retrieve_mode,
        'retrieve_local': retrieve_local,
        'retrieve_local_size': retrieve_local_size, 
        'fattn': True,
        'block_size': n_frame_tokens,
        'retrieve_size': retrieve_size,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = LlavaOneVision_StreamKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        dataset=dataset,
        guidance_prompt_ids=guidance_prompt_ids,
        compression_ratio=compression_ratio,
        compress_mode=compress_mode,
        compress_temp=compress_temp,
        retrieve_mode=retrieve_mode,
        retrieve_temp=retrieve_temp,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        segment_theta=segment_theta,
        segment_mode=segment_mode,
        segment_summary_sign=segment_summary_sign,
        n_local=n_local,
        retrieve_size=retrieve_size,
        chunk_size=chunk_size,
    )
    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    model.eval()

    return model, processor
