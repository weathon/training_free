# https://github.com/huggingface/diffusers/blob/01abfc873659e29a8d002f20782fa5b5e6d03f9c/src/diffusers/models/attention_processor.py#L930
import torch.nn.functional as F
from typing import Optional
import torch
import math

class MochiAttnProcessor2_0:
    """Attention processor used in Mochi."""

        
    def __init__(self, token_index_of_interest: torch.Tensor, positive_mask: torch.Tensor):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MochiAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

        self.token_index_of_interest = token_index_of_interest
        self.positive_mask = positive_mask
        
        
    def __call__(
        self,
        attn: "MochiAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states = encoder_hidden_states.detach()
        query = attn.to_q(hidden_states).detach()
        key = attn.to_k(hidden_states).detach()
        value = attn.to_v(hidden_states).detach()

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)

            query = apply_rotary_emb(query, *image_rotary_emb)
            key = apply_rotary_emb(key, *image_rotary_emb)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        encoder_query, encoder_key, encoder_value = (
            encoder_query.transpose(1, 2),
            encoder_key.transpose(1, 2),
            encoder_value.transpose(1, 2),
        )

        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)
        total_length = sequence_length + encoder_sequence_length

        batch_size, heads, _, dim = query.shape
        attn_outputs = []
        for idx in range(batch_size):
            mask = attention_mask[idx][None, :]
            valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

            valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

            valid_query = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
            valid_key = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
            valid_value = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)

            attn_output = F.scaled_dot_product_attention(
                valid_query, valid_key, valid_value, dropout_p=0.0, is_causal=False
            )
            
            valid_sequence_length = attn_output.size(2)
            attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
            attn_outputs.append(attn_output)


        # # should we switch from image atten to the text? that makes more sense 
        # # https://arxiv.org/pdf/2408.14826 use image as query, text as key
        # # print(encoder_query.shape, valid_encoder_query.shape, valid_prompt_token_indices)
        # interested_query = encoder_query[0,:,self.token_index_of_interest]
        # # interested_query = valid_encoder_query[0,:,self.token_index_of_interest]
        # image_keys = key[0]
        # attention_scores = torch.einsum('hqd,hkd->hqk', interested_query, image_keys).unsqueeze(0)
        # # should not softmax like this, because the sequence length includes the time dim? it is softmax on key dim
        # self.attn_weights = attention_scores
        # # self.attn_weights = F.softmax(attention_scores / math.sqrt(interested_query.size(-1)), dim=-1)

        print(encoder_key.shape)
        interested_key = encoder_key[1, :, self.token_index_of_interest]
        image_queries = query[1]
        attention_scores = torch.einsum('hqd,hkd->hqk', image_queries, interested_key).unsqueeze(0)
        # self.attn_weights = F.softmax(attention_scores / math.sqrt(image_queries.size(-1)), dim=-1).permute(0, 1, 3, 2)
        self.attn_weights = attention_scores.permute(0, 1, 3, 2)
        # self.attn_weights = attention_scores.permute(0, 1, 3, 2) #(attention_scores - attention_scores.min(axis=-1, keepdims=True)) / (attention_scores.max(axis=-1, keepdims=True) - attention_scores.min(axis=-1, keepdims=True))
        
        
        
        hidden_states = torch.cat(attn_outputs, dim=0)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
