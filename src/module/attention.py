"""
Low precision attention modules
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from src.module.base import _QBase, _QBaseLinear, IntMatMul
from src.module.fuse import MulShift

class QAttention(nn.Module):
    def __init__(
            self,
            dim:int,
            num_heads, 
            qkv_bias=False,
            qk_norm=False, 
            attn_drop=0.0,
            proj_drop=0.0,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert dim % num_heads == 0,"dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # attention scale
        self.scale = self.head_dim ** (-0.5)

        self.qkv = _QBaseLinear(dim, int(dim*3), bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = _QBaseLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_scale = MulShift()
        self.qkv_deq = MulShift()

        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(self.scale)

        # quantizers 
        self.xq = _QBase(nbit=32)
        self.qqkv = _QBase(nbit=32)
        self.qproj = _QBase(nbit=32)

        # training flag
        self.train_flag = True

        # matmul operator
        self.qk = IntMatMul(nbit=32)
        self.attnv = IntMatMul(nbit=32)
    
    def inference(self):
        self.train_flag = False
        
        self.qqkv.inference()
        self.qkv.wq.inference()
        self.xq.inference()
        self.qkv.inference()
        self.qproj.inference()
        self.proj.inference()

    def trainFunc(self, q, k, v):
        attn = q @ k.transpose(-2, -1)  # out dim = token x token
        attn = self.attn_scale(attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v                    # out dim = token x head_dim
        x = self.qproj(x)

        return x
    
    def evalFunc(self, q, k, v):
        q, k, v = q.double(), k.double(), v.double()

        attn = self.qk(q, k.transpose(-2, -1))
        attn = self.attn_scale(attn)

        attn = F.softmax(attn, dim=-1)
        attn = attn.mul(255.).round()

        attn = self.attn_drop(attn)

        x = self.attnv(attn, v)
        x = self.qproj(x)

        return x.float()
    
    def forward(self, x:torch.Tensor):
        B, N, C = x.shape

        x = self.xq(x)
        qkv = self.qkv(x)
        qkv = self.qqkv(qkv)

        # reshape
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # reshape to (qkv), batch, num_heads, token, head_dim
        
        q, k, v = qkv.unbind(0)         # batch, num_heads, token, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        if self.train_flag:
            x = self.trainFunc(q, k, v)
        else:
            x = self.evalFunc(q, k, v)

        # reshape
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.qkv_deq(x)
        x = self.proj_drop(x)
        return x
    
class QBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.t2c_init(config)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def t2c_init(self, config):
        # quantizers
        self.xq = _QBase(nbit=32)
        self.qquery = _QBase(nbit=32)
        self.qkey = _QBase(nbit=32)
        self.qvalue = _QBase(nbit=32)
        
        self.qkv_deq = MulShift()
        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(1 / math.sqrt(self.attention_head_size))

        # train flag
        self.train_flag = True

    def inference(self):
        self.xq.inference()
        self.qquery.inference()
        self.qkey.inference()
        self.qvalue.inference()

        self.query.inference()
        self.key.inference()
        self.value.inference()

        self.train_flag = False

        # matmul
        self.qk = IntMatMul(nbit=32)
        self.attnv = IntMatMul(nbit=32)
    
    def trainFunc(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
            
        hidden_states = self.xq(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        # print(mixed_query_layer.mean().item())

        # low precision Q
        mixed_query_layer = self.qquery(mixed_query_layer)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key = self.key(encoder_hidden_states)
            key_layer = self.qkey(key)
            key_layer = self.transpose_for_scores(key)

            value = self.value(encoder_hidden_states)
            value_layer = self.qvalue(value)
            value_layer = self.transpose_for_scores(value)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key = self.key(hidden_states)
            key_layer = self.qkey(key)
            key_layer = self.transpose_for_scores(key_layer)

            value = self.value(hidden_states)
            value_layer = self.qvalue(value)
            value_layer = self.transpose_for_scores(value_layer)
            
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key = self.key(hidden_states)
            key_layer = self.qkey(key)
            key_layer = self.transpose_for_scores(key_layer)

            value = self.value(hidden_states)
            value_layer = self.qvalue(value)
            value_layer = self.transpose_for_scores(value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print(attention_scores.mean().item())

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = self.attn_scale(attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.qkv_deq(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def evalFunc(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        hidden_states = self.xq(hidden_states)

        # Q
        mixed_query_layer = self.query(hidden_states)
        mixed_query_layer = self.qquery(mixed_query_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # K
        key = self.key(hidden_states)
        key_layer = self.qkey(key)
        key_layer = self.transpose_for_scores(key_layer)

        # V
        value = self.value(hidden_states)
        value_layer = self.qvalue(value)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.qk(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.attn_scale(attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # round the attention score to 8-bit (fixed)
        attention_probs = attention_probs.mul(256.).round()

        context_layer = self.attnv(attention_probs, value_layer)
        context_layer = self.qkv_deq(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        if self.train_flag:
            output = self.trainFunc(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions
            )
        else:
            output = self.evalFunc(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions
            )

        return output