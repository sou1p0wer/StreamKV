import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger
from model.patch import patch_hf
from model.abstract_streamkv import Abstract_StreamKV, obtain_cdf_num, rescale_sum_match

class LlavaOneVision_StreamKV(LlavaOnevisionForConditionalGeneration, Abstract_StreamKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, dataset, encode_prompt_ids, compression_ratio, compress_temp, retrieve_temp, max_segment_size, min_segment_size, segment_theta, segment_mode, use_segment_summary, n_local, retrieve_size, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_StreamKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, dataset, encode_prompt_ids, compression_ratio, compress_temp, retrieve_temp, max_segment_size, min_segment_size, segment_theta, segment_mode, use_segment_summary, n_local, retrieve_size, chunk_size)

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
            for layer_kv in self.kv_cache:
                layer_kv.set_retrieval_prefill()

            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            retrieval_score = []
            for layer_kv in self.kv_cache:
                layer_kv.reset_retrieval_prefill()
                retrieval_score.append(layer_kv.layer_retrieval_score)

            target_num = len(self.kv_cache) * self.retrieve_size
            retrieval_nums = obtain_cdf_num(retrieval_score, target_num)
            retrieval_nums = rescale_sum_match(retrieval_nums,max(self.retrieve_size*(1-self.retrieve_temp),1),min(self.retrieve_size*(1+self.retrieve_temp),128))
            for i, layer_kv in enumerate(self.kv_cache):
                layer_kv.retrieval_num = retrieval_nums[i]

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

        output_ids = []
        stopped = False
        
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


def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_encode=None, n_local=None, dataset=None, compression_ratio=None, compress_temp=None, retrieve_temp=None, retrieve_local=None, retrieve_local_size=None, max_segment_size=None, min_segment_size=None, segment_theta=None, use_segment_summary=None, segment_mode=None, retrieve_size=64, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant. Please actively track and remember current objects, actions, on-screen texts/numbers, temporal changes, and any anomalies. Focus on capturing key visual clues, filtering misleading or outdated context, and keep in mind prior key events to support sequential and time-critical queries or proactive outputs throughout the streaming video.<|im_end|><|im_start|>user '
    encode_prompt = 'For this event segment in the video, please prioritize retaining the following visual information:1)Key objects, people, and their attributes, and any changes or transitions during the event.2)Major actions, state transitions, and interactions (e.g., who/what is doing what, when, and where), including the precise timing of key moments.3)On-screen text, numbers, and all visible cues (such as labels, scores, timer, signs, etc.).4)Causal relationships, outcome(s), anomalies, and salient shifts within the segment (e.g., cause-effect chains, rare or surprising events, turning points).5)Counts and spatial arrangements of important objects or people, as well as layout details that may support summarization, prediction, or sequential reasoning in later queries.6)Any misleading, ambiguous, or easily confused visual clues—ensure enough context is preserved to distinguish the current event state from similar past/future states.'
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    encode_prompt_ids = processor.tokenizer(encode_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_encode': encode_prompt_ids.shape[1] if n_encode is None else n_encode,
        'use_segment_summary': use_segment_summary, 
        'n_local': n_local,
        'compression_ratio': compression_ratio,
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
        encode_prompt_ids=encode_prompt_ids,
        compression_ratio=compression_ratio,
        compress_temp=compress_temp,
        retrieve_temp=retrieve_temp,
        max_segment_size=max_segment_size,
        min_segment_size=min_segment_size,
        segment_theta=segment_theta,
        segment_mode=segment_mode,
        use_segment_summary=use_segment_summary,
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
