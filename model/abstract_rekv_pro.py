import torch
from logzero import logger
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

def cossin_similarity2(a, b):
    a = a.squeeze(0) #
    b = b.squeeze(0)
    a = a.mean(dim=0,keepdim=False)
    b = b.mean(dim=0,keepdim=False)
    dot_product = torch.dot(a, b)
    norm_product = torch.norm(a) * torch.norm(b)
    return dot_product / norm_product

# def cossin_similarity3(a, b):
#     #(196,196)的对角线 3个一组
#     return dot_product

# def cossin_similarity4(a, b):
#     #(196,196)的对角线 超过某一个阈值的比例
#     return dot_product

def merge(cur_chunk_feature,cur_chunk_cossim):
    max_cossim = max(cur_chunk_cossim)
    max_cossim_idx = cur_chunk_cossim.index(max_cossim)
    pre_feature_idx = max_cossim_idx
    post_feature_idx = max_cossim_idx+1
    pre_feature = cur_chunk_feature[pre_feature_idx]
    post_feature = cur_chunk_feature[post_feature_idx]
    merged_feature = (pre_feature+post_feature)/2
    #更新cur_chunk_cossim
    if pre_feature_idx == 0:
        cos_sim = cossin_similarity2(merged_feature,cur_chunk_feature[post_feature_idx+1])
        cur_chunk_cossim.pop(0)
        cur_chunk_cossim[0] = cos_sim
    elif post_feature_idx == len(cur_chunk_feature)-1:
        cos_sim = cossin_similarity2(merged_feature,cur_chunk_feature[pre_feature_idx-1])
        cur_chunk_cossim.pop(-1)
        cur_chunk_cossim[-1] = cos_sim
    else:
        cos_sim_pre = cossin_similarity2(merged_feature,cur_chunk_feature[pre_feature_idx-1])
        cos_sim_post = cossin_similarity2(merged_feature,cur_chunk_feature[post_feature_idx+1])
        cur_chunk_cossim[max_cossim_idx - 1] = cos_sim_pre
        cur_chunk_cossim[max_cossim_idx + 1] = cos_sim_post
        cur_chunk_cossim.pop(max_cossim_idx)
    #更新cur_chunk_feature
    cur_chunk_feature[pre_feature_idx] = merged_feature
    cur_chunk_feature.pop(post_feature_idx)

    return cur_chunk_feature,cur_chunk_cossim

class Abstract_ReKV:
    processor = None
    kv_cache = None #kv_cache_manager

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, encode_prompt_ids, compression_ratio, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.encode_prompt_ids = encode_prompt_ids
        self.compression_ratio = compression_ratio
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self): #编码初始prompt
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        #print(self.init_prompt_ids)
        #print(self.init_prompt_ids.shape) ([1,13])
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values #type ContextManager

    def _get_video_features(self, pixel_values_videos): #实现由子类提供 (B, F, C, H, W) -> (B, Nv*196, D)
        pass

    def _encode_video_chunk(self, video_features): #编码视频片段,只为更新kv_cache
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'
        if self.compression_ratio < 1:
            if not isinstance(self.init_prompt_ids, torch.Tensor):
                self.encode_prompt_ids = torch.as_tensor([self.encode_prompt_ids], device=self.device)
            output = self.language_model(input_ids=self.encode_prompt_ids, inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True) #对应patch.py里的model_forward
        else:
            output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values #type ContextManager
    
    @torch.inference_mode()
    def encode_video(self, video, video_path=None, min_chunk_size=8, max_chunk_size=64, sim_theta=0.95):  # video: (Nv, H, W, 3)
        log_file_path = '/home/cyl476530/ReKV/encode.txt'
        # encode chunk by chunk，目标更新整个video的kv_cache
        num_frames = video.shape[0]
        cur_chunk_feature = []
        cur_chunk_cossim = []
        cur_chunk_size = 0
        chunk_start_idx = 0
        with open(log_file_path, 'a') as f:
            #f.write('-----------------------start-------------------------\n')
            #f.write(f'video_path: {video_path}\n')
            for frame_idx in range(num_frames):
                cur_frame = video[frame_idx:frame_idx+1]
                pixel_values_frame = self.processor.video_processor(cur_frame, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
                cur_frame_feature = self._get_video_features(pixel_values_frame)  #(1, 196, D)
                if not cur_chunk_feature:
                    cur_chunk_feature.append(cur_frame_feature)
                    past_frame_feature = cur_frame_feature
                    cur_chunk_size = 1
                    continue
                cos_sim = cossin_similarity2(cur_frame_feature,past_frame_feature)
                cur_chunk_cossim.append(cos_sim)
                if cur_chunk_size < min_chunk_size or cos_sim > sim_theta:
                    cur_chunk_feature.append(cur_frame_feature)
                    if cur_chunk_size >= max_chunk_size:
                        cur_chunk_feature,cur_chunk_cossim  = merge(cur_chunk_feature,cur_chunk_cossim)
                    else:
                        cur_chunk_size += 1
                    past_frame_feature = cur_frame_feature
                else:
                    #f.write(f'len(cur_chunk_feature): {len(cur_chunk_feature)}\n')
                    cur_chunk_tensor = torch.cat(cur_chunk_feature, dim=1)
                    self._encode_video_chunk(cur_chunk_tensor)

                    #f.write(f'[{chunk_start_idx * 2}s,{(frame_idx - 1) * 2}s]')
                    #print(f"Chunk processed from frame {chunk_start_idx} to frame {frame_idx - 1}")
                    chunk_start_idx = frame_idx
                    cur_chunk_feature = [cur_frame_feature]
                    past_frame_feature = cur_frame_feature
                    cur_chunk_cossim = []
                    cur_chunk_size = 1
            #f.write(f'len(cur_chunk_feature): {len(cur_chunk_feature)}\n')
            cur_chunk_tensor = torch.cat(cur_chunk_feature, dim=1)
            self._encode_video_chunk(cur_chunk_tensor)
            #f.write('-----------------------end-------------------------\n')
            #f.write(f'[{chunk_start_idx * 2}s,{(frame_idx - 1) * 2}s]\n')


    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128): #实现由子类提供
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory
