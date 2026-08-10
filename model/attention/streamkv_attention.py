import copy
import torch
from typing import Optional

from .kv_cache_manager import ContextManager
from .dot_production_attention import get_multi_stage_dot_production_attention

def streamkv_attention_forward(
    n_local, n_init, n_encode, use_segment_summary, retrieve_size, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    compression_ratio,
    retrieve_local, retrieve_local_size,
    async_global_stream=True,
    pin_memory=False,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self,query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
    ):

        """ 1. Project QKV """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)
        h_k = project_k(key_value)
        h_v = project_v(key_value)
        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()

        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                n_init, n_local, use_segment_summary,
                block_size, max_cached_block, retrieve_size, chunk_size, exc_block_size,
                retrieve_local, retrieve_local_size,
                fattn,
                async_global_stream,
                pin_memory,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
            if type(past_key_value) is ContextManager:
                past_k, past_v = past_key_value.get_retrieved_kv(global_q)
                update_kv_cache = False
            else:
                past_k = past_key_value[0]
                past_v = past_key_value[1]
                update_kv_cache = True

            """ 2. Update KV w/ past KV cache """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]
            """ 3. Update KV cache """
            if update_kv_cache:
                h_k_cache = h_k
                h_v_cache = h_v
                current_key_value = (h_k_cache, h_v_cache)
            else:
                current_key_value = (past_k, past_v)

            """ 4. Get local QKV and apply RoPE to local QK """
            h_q_, h_k_, h_v_ = h_q, h_k, h_v

            local_h_q, local_h_k = position_bias(h_q_, h_k_)
            local_h_v = h_v_

            init_h_q = h_q
            init_h_k = torch.empty(
                (batch_size, num_heads_kv, 0, dim_head),
                device=h_k.device,
                dtype=h_k.dtype
            )
            init_h_v = torch.empty(
                (batch_size, num_heads_kv, 0, dim_head),
                device=h_v.device,
                dtype=h_v.dtype
            )

            """ 6. Sliding Window Attention """
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=9999999)
            attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, 9999999), complement_sliding_window=True)
            score, _ = attn.get_result()

            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            score = score.reshape(batch_size, len_q, num_heads * dim_head)
            score = attention_out(score)

            return score, current_key_value

        else:
            if past_key_value.initialized and compression_ratio < 1:
                video_len = len_q - n_encode
                video_local_q, video_local_k, video_local_v = local_q[:,:,:video_len,:], local_k[:,:,:video_len,:], local_v[:,:,:video_len,:]
                video_global_q, video_global_k, video_global_v = global_q[:,:,:video_len,:], global_k[:,:,:video_len,:], global_v[:,:,:video_len,:]
                prompt_local_q, prompt_local_k, prompt_local_v = local_q[:,:,video_len:,:], local_k[:,:,video_len:,:], local_v[:,:,video_len:,:]
                prompt_global_q, prompt_global_k, prompt_global_v = global_q[:,:,video_len:,:], global_k[:,:,video_len:,:], global_v[:,:,video_len:,:]
                video_o = past_key_value.append(
                    video_local_q, video_local_k, video_local_v,
                    video_global_q, video_global_k, video_global_v,
                )
                video_o = video_o.view(batch_size, num_heads, video_len, dim_head).permute(0, 2, 1, 3)
                video_o = video_o.reshape(batch_size, video_len, dim_head * num_heads)
                video_o = attention_out(video_o)

                prompt_q = h_q[:,:,video_len:,:]
                local_h_q, local_h_k = position_bias(prompt_q, h_k)
                local_h_v = h_v
                attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
                attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local, end=True)
                prompt_o, _ = attn.get_result()
                prompt_o = prompt_o.view(batch_size, num_heads, n_encode, dim_head).permute(0, 2, 1, 3) 
                prompt_o = prompt_o.reshape(batch_size, n_encode, num_heads * dim_head)
                prompt_o = attention_out(prompt_o)
                o = torch.cat([video_o, prompt_o], dim=1)

                past_key_value.calculate_score(video_len, prompt_global_q)
                
            else:
                o = past_key_value.append(
                    local_q, local_k, local_v,
                    global_q, global_k, global_v,
                )
                o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
                o = o.reshape(batch_size, len_q, dim_head * num_heads)
                o = attention_out(o)

            return o, past_key_value

    return forward
