# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch GIT model."""


import math
import os
import os.path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import random 
from torch.nn import functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_git import GitConfig, GitVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/git-base"
_CONFIG_FOR_DOC = "GitConfig"

GIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/git-base",
    # See all GIT models at https://huggingface.co/models?filter=git
]

TOT_LEN_HARDCODED=1

def print_module_stats(ss,mm) :
    return;
    print("---- " + str(ss) + " ----")
    print(mm.shape)
    print("std "+str(mm.std().item()) + " mean " + str(mm.mean().item()) + " min " + str(mm.min().item()) + " max " + str(mm.max().item()) )


@dataclass
class GitVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    
class GitEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GitSelfAttention(nn.Module):
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
        #self.image_patch_tokens = int((config.vision_config.image_size / config.vision_config.patch_size) ** 2 + 1)
        self.image_patch_tokens = TOT_LEN_HARDCODED 
        if config.num_image_with_embedding is not None:
            self.image_patch_tokens *= config.num_image_with_embedding

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

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        cutoff = self.image_patch_tokens if pixel_values_present else 0
        if past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([key_layer[:, :, :cutoff, :], past_key_value[0], key_layer[:, :, -1:, :]], dim=2)
            value_layer = torch.cat(
                [value_layer[:, :, :cutoff, :], past_key_value[1], value_layer[:, :, -1:, :]], dim=2
            )
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        # NOTE: like in other caches, we store the text component. In GIT it means we discard the image component.
        past_key_value = (
            key_layer[:, :, cutoff:, :],
            value_layer[:, :, cutoff:, :],
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

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

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in GitModel forward() function)
            # print("   =>" + str(attention_scores.shape))
            # print("   =>" + str(attention_mask.shape))
            attention_scores = attention_scores + attention_mask
        #import pdb; pdb.set_trace()   

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class GitSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GitAttention(nn.Module):
    # Copied from transformers.models.bert.modeling_bert.BertAttention.__init__ with Bert->Git
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = GitSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = GitSelfOutput(config)
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value,
            output_attentions,
            pixel_values_present,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class GitIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class GitOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GitLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GitAttention(config)
        self.intermediate = GitIntermediate(config)
        self.output = GitOutput(config)
#        import pdb; pdb.set_trace()            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            pixel_values_present=pixel_values_present,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class GitEncoder(nn.Module):
    # Copied from transformers.models.bert.modeling_bert.BertEncoder.__init__ with Bert->Git
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([GitLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                    pixel_values_present,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GitConfig
    base_model_prefix = "git"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        """
        if isinstance(module, GitVisionEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.patch_embedding.weight, std=self.config.initializer_range)
            nn.init.normal_(module.position_embedding.weight, std=self.config.initializer_range)
        """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        #if isinstance(module, (GitEncoder, GitVisionEncoder)):
        if isinstance(module, (GitEncoder)):    
            module.gradient_checkpointing = value


GIT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Copied from transformers.models.clip.modeling_clip.CLIPMLP
class GitVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPAttention
class GitVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# here was gitvisonencoder

class GitLidarModel(GitPreTrainedModel):
    main_input_name = "points"

    # Copied from transformers.models.clip.modeling_clip.CLIPVisionModel.__init__ with CLIP->Git
    def __init__(self,config: GitConfig):
        #self.vision_model = GitVisionTransformer(config)
        super().__init__(config)
        self.lidar_model = None
        self.sop = None
        self.use_sop = False
        self.root_path = "/tmp/"
        #self.lidar_projection = nn.Conv2d(in_channels=128,out_channels=config.hidden_size,kernel_size=(32,32), stride=(32,32))
        
        self.lidar_projection = nn.Conv2d(in_channels=128,out_channels=config.hidden_size,kernel_size=(14,14), stride=(14,14)).to(dtype=torch.float32)
        self.lidar_projection2 = nn.Conv2d(in_channels=128,out_channels=config.hidden_size,kernel_size=(20,20), stride=(20,20)).to(dtype=torch.float32)
        self.bt_norm = nn.BatchNorm2d(config.hidden_size, eps=1e-3, momentum=0.01)
        self.bt_norm2d = nn.BatchNorm2d(config.hidden_size, eps=1e-4, momentum=0.1)
        self.post_layernorm = nn.LayerNorm(int(config.hidden_size/3), eps=config.layer_norm_eps).to(dtype=torch.float32)
        #self.rl = nn.ReLU(inplace=True) 


    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding


    def voxel_sop(self,ret_dict) :
        lid_mask = ret_dict['voxel_mae_mask']
        bb_filter = lid_mask!=0
        lid_mask = ret_dict['voxel_mae_mask'][bb_filter]
        lid_feat = ret_dict['voxel_features'][bb_filter]                
        vox_coords = ret_dict['voxel_coords'][bb_filter]
        max_len = torch.bincount(vox_coords[:,0]).max().item()
        lid_feat_vec = None
        lid_mask_vec = None


        for ii in range(batch_size) :
            bm = vox_coords[:,0] == ii
            lid_feat_ii = lid_feat[bm,:]
            lid_mask_ii = lid_mask[bm]
            target_feat = torch.zeros(max_len, lid_feat_ii.shape[1]).to(projected_visual_features.device)
            target_mask = torch.zeros(max_len).to(projected_visual_features.device)
            target_feat[:lid_feat_ii.shape[0],:] = lid_feat_ii
            target_mask[:lid_mask_ii.shape[0]] = lid_mask_ii

            if ii == 0 :
                lid_feat_vec = target_feat[None,:,:]
                lid_mask_vec = target_mask[None,:]
            else :
                lid_feat_vec = torch.cat([lid_feat_vec,target_feat[None,:,:]])
                lid_mask_vec = torch.cat([lid_mask_vec,target_mask[None,:]])

        return self.sop(lid_feat_vec)



    def save_tensor(self,xx,nn,lab) :
        acc = 0
        for ll in xx :
            fname =  os.path.splitext(nn[acc])[0]+'.pt'
            torch.save(ll,self.root_path / lab  / fname)
            acc = acc + 1


    def load_pickle(self,nn,lab) :
        xx = []
        for ll in nn :
            fname = self.root_path /  lab /  (os.path.splitext(ll)[0]+'.pt')
            if os.path.isfile(fname) :  
               xx1 = torch.load(fname)
               #import pdb; pdb.set_trace()                                
               xx.append(random.choice(xx1))
            else :
                print("TENSOR NOT FOUND")
                print(fname)
                return None
        return torch.stack(xx)
            
    def load_tensor(self,nn,lab) :
        xx = []
        for ll in nn :
            fname = self.root_path /  lab /  (os.path.splitext(ll)[0]+'.pt')
            if os.path.isfile(fname) : 
                xx1 = torch.load(fname)
                xx.append(xx1)
            else :
                print("TENSOR NOT FOUND")
                print(fname)
                return None
        return torch.stack(xx)

        
            
    def forward(
        self,
        points: Optional[torch.Tensor] = None,            
        lidar_values: Optional[dict] = None,
        input_mod: Optional[str] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Returns:
        ```"""
        #return_dict_v2 = return_dict if return_dict is not None else self.config.use_return_dict
        #lidar_values['points'] = points
        
        use_log3dnet = False
        use_cache_sop = True
        use_cache_gdmae = True
        use_positional = False
        use_batch_norm = False
        use_duplicate_vect = False
        
        em_len=int((TOT_LEN_HARDCODED)*3+1)

        ##################################################################################  
        # load desc from loggnet
        ##################################################################################    
        desc_from_logg = False 
        desc_from_logg = True
        
        if desc_from_logg == True:

            output_desc_computed = lidar_values['desc'][0]

            
            """
            from .logg3d_net_desc import get_logg3d_net_desc
            eval_seq = '06'
            output_desc_computed = get_logg3d_net_desc(eval_seq, lidar_values, input_mod, voxel_size=0.1)
            """
    
            """
            
            import numpy as np
            from models.pipeline_factory import get_pipeline
            #from config.eval_config import get_config_eval
            from models.pipelines.pipeline_utils import make_sparse_tensor
            import random
             
            #cfg = get_config_eval()
            voxel_size = 0.1
            # Get model
            
            model = get_pipeline('LOGG3D')

            if eval_seq == '00' or eval_seq == '22' :
                save_path =  "/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth"
                
            elif eval_seq == '02':
                save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_05-55-20_3n24h_Kitti_v10_q29_10s2_262448.pth'
            elif eval_seq == '05':
                save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_06-11-58_3n24h_Kitti_v10_q29_10s5_262449.pth'
                
            elif eval_seq == '06':
                save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_06-43-47_3n24h_Kitti_v10_q29_10s6_262450.pth'
            elif eval_seq == '07':
                save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_08-34-46_3n24h_Kitti_v10_q29_10s7_262451.pth'
            elif eval_seq == '08':
                save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth'

                
            checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
            model.load_state_dict(checkpoint['model_state_dict'])

            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        
            model = model.cuda()
            model.eval()

            def random_rotate(xyzr, r_angle=360, is_random=True, add_noise=True, rand_tr=False):
                # If is_random = True: Rotate about z-axis by random angle upto 'r_angle'.
                # Else: Rotate about z-axis by fixed angle 'r_angle'.
                r_angle = (np.pi/180) * r_angle
                if is_random:
                    r_angle = r_angle*np.random.uniform()
                cos_angle = np.cos(r_angle)
                sin_angle = np.sin(r_angle)
                rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                                    [sin_angle, cos_angle, 0],
                                    [0,         	0,  	1]])
                scan = xyzr[:, :3]
                int = xyzr[:, 3].reshape((-1, 1))
                augmented_scan = np.dot(scan, rot_matrix)
                
                if add_noise:
                    n_sigma = 0.01  # Add gaussian noise
                    noise = np.clip(n_sigma * np.random.randn(*
                                    augmented_scan.shape), -0.03, 0.03)
                    augmented_scan = augmented_scan + noise
                
                if rand_tr:
                    tr_xy_max, tr_z_max = 1.5, 0.25
                    tr_xy = np.clip(np.random.randn(1, 2), -tr_xy_max, tr_xy_max)
                    tr_z = np.clip(0.1*np.random.randn(1, 1), -tr_z_max, tr_z_max)
                    tr = np.hstack((tr_xy, tr_z))
                    augmented_scan = augmented_scan + tr
                
                augmented_scan = np.hstack((augmented_scan, int))
                return augmented_scan.astype(np.float32)
                
            def occlude_scan(scan, angle=30):
                # Remove points within a sector of fixed angle (degrees) and random heading direction.
                thetas = (180/np.pi) * np.arctan2(scan[:, 1], scan[:, 0])
                heading = (180-angle/2)*np.random.uniform(-1, 1)
                occ_scan = np.vstack(
                    (scan[thetas < (heading - angle/2)], scan[thetas > (heading + angle/2)]))
                return occ_scan.astype(np.float32)
            
            
            lidar_file = '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/'+eval_seq+'/velodyne/' + lidar_values[input_mod][0][1]
            lidar_pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

            random_rotation = False
            random_occlusion = False
            random_scale = False
            max_scale = 1.2
            min_scale = 0.8
            
            lidar_pc2 = lidar_pc
            if random_rotation:
                lidar_pc2 = random_rotate(lidar_pc2)
            if random_occlusion:
                lidar_pc2 = occlude_scan(lidar_pc2)
            if random_scale and random.random() < 0.95:
                scale = min_scale + \
                    (max_scale - min_scale) * random.random()
                lidar_pc2 = scale * lidar_pc2

            input2 = make_sparse_tensor(lidar_pc2, voxel_size).cuda()
            output_desc2, output_feats2 = model(input2)  # .squeeze()
            #output_feats2 = output_feats2[0]
            #global_descriptor2 = output_desc2.cpu().detach().numpy()
        """
        
        ##################################################################################  
        # load desc
        ##################################################################################   
        ret_dict = lidar_values
        if desc_from_logg == False:
            nn = lidar_values[input_mod][:,1] # change 'frame_id' to input_mod
            xx_sop_log3d = self.load_tensor(nn,"logg_desc")
        else:
           xx_sop_log3d = output_desc_computed.reshape([1, -1]) 
            
        #import pdb; pdb.set_trace()

        
        #xx_final_ori = xx_sop_log3d.unsqueeze(1).repeat(1, 1, 1).to(dtype=torch.float32)
        xx_final_ori = xx_sop_log3d.unsqueeze(1).to(dtype=torch.float32)
        del xx_sop_log3d
        print_module_stats("xx_final_ori",xx_final_ori)
        if use_batch_norm : 
            xx_final_ori = self.post_layernorm(xx_final_ori)
        print_module_stats("xx_final_ori_norm", xx_final_ori)


        ##################################################################################  
        # desc dim 256 to 768
        ##################################################################################   
        for ii in range(0,em_len) : # (0, 4)
            #denominator = np.power(10000, 2*ii/em_len)
            pii = math.floor(ii/2)
            if use_positional : 
                if use_duplicate_vect :
                     xx_embed = xx_final_ori
                
                elif ii % 2 == 0 :
                    xx_embed = torch.sin(math.pow(2,pii)*xx_final_ori)
                else :
                    xx_embed = torch.cos(math.pow(2,pii)*xx_final_ori)
            else :
                xx_embed = xx_final_ori
                
            if ii % 3 == 0 : # si dernier passage
                embed_stack = xx_embed
            else :
                embed_stack = torch.cat([embed_stack,xx_embed],dim=2)

            if ii % 3 == 2 :
                if ii == 2 :
                    xx_final = embed_stack
                else :
                    xx_final = torch.cat([xx_final,embed_stack],dim=1)
                                                  
        print_module_stats("xx_final_full" + str(ii),xx_final)                                              
        xx_final = xx_final.to(dtype=torch.float32)
        return (ret_dict,xx_final)
 
    
@add_start_docstrings(
    """The vision model from CLIP, used in GIT, without any head or projection on top.""",
    GIT_START_DOCSTRING,
)


@add_start_docstrings(
    "The bare GIT Model transformer consisting of a CLIP image encoder and text decoder outputting raw hidden-states"
    " without any specific head on top.",
    GIT_START_DOCSTRING,
)
class GitModel(GitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = GitEmbeddings(config)
        #self.image_encoder = GitVisionModel(config.vision_config)
        self.lidar_encoder = GitLidarModel(config)
        self.encoder = GitEncoder(config)
        self.num_feat_lidar = -1
        #self.visual_projection = GitProjection(config)

        if config.num_image_with_embedding is not None:
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size))
                for _ in range(config.num_image_with_embedding)
            )
        #import pdb; pdbset_trace()
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _generate_future_mask(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def create_attention_mask(self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=None):
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)

        top_right = torch.full(
            (num_memory, num_tgt + past_key_values_length),
            float("-inf"),
            device=tgt.device,
            dtype=dtype,
        )
        bottom_left = torch.zeros(
            (num_tgt, num_memory),
            dtype=dtype,
            device=tgt_mask.device,
        )

        if past_key_values_length > 0:
            tgt_mask = torch.zeros(
                (tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length),
                dtype=dtype,
                device=tgt_mask.device,
            )

        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # if it is False, it means valid. That is, it is not a padding
        if memory_key_padding_mask.dtype != torch.bool:
            raise ValueError("Memory key padding mask must be a boolean tensor.")
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float("-inf")
        full_attention_mask = full_attention_mask.expand(
            (memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + past_key_values_length + num_tgt)
        )
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]

        return full_attention_mask

    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        lidar_values: Optional[dict] = None,
        input_mod: Optional[str] = None,
        text : Optional[dict] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> import requests
        >>> from PIL import Image

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base")
        >>> model = AutoModel.from_pretrained("microsoft/git-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = "this is an image of two cats"

        >>> inputs = processor(text, images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        projected_visual_features = None
        
        ret_dict,projected_lidar_features = self.lidar_encoder(points,lidar_values,input_mod) # load
        projected_visual_features=projected_lidar_features # torch.Size([16, 255, 768])
        batch_size = ret_dict['batch_size']


        tot_len = TOT_LEN_HARDCODED 
        projected_visual_features = projected_lidar_features[:,:tot_len,:]
        self.num_feat_lidar = projected_visual_features.shape[1]

        #print(input_ids)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if projected_visual_features is None:
            projected_visual_features = torch.zeros(
                (embedding_output.shape[0], 0, embedding_output.shape[2]),
                dtype=embedding_output.dtype,
                device=embedding_output.device,
            )


        # Repeat visual features to match embedding batch size.
        projected_visual_features = torch.repeat_interleave(projected_visual_features,embedding_output.size(0) // projected_visual_features.size(0), dim=0)

        # concatenate patch token and text token embeddings
        hidden_states = torch.cat((projected_visual_features, embedding_output), dim=1)

        # By default, an additive causal mask is created
        # for masking the future (one direction).
        tgt_mask = self._generate_future_mask(seq_length, embedding_output.dtype, embedding_output.device)

        # Create an attention mask of shape (batch_size, 1, tgt_seq_len, src_seq_len)
        combined_attention_mask = self.create_attention_mask(
            tgt=embedding_output,
            memory=projected_visual_features,
            tgt_mask=tgt_mask,
            past_key_values_length=past_key_values_length,
        )

        
        if attention_mask is not None:
            # if the user provides an attention mask, we add it to the default one
            expanded_attn_mask = _expand_mask(attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]).to(
                embedding_output.device
            )
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[:, :, -past_key_values_length:, :]
            else:                                                  
                print_module_stats("exp mask",expanded_attn_mask)
                print_module_stats("cb  mask",combined_attention_mask[:, :, -input_shape[1] :, -input_shape[1] :])            
                combined_attention_mask[:, :, -input_shape[1] :, -input_shape[1] :] += expanded_attn_mask
                
        
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=combined_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values_present=((points is not None) or (pixel_values is not None)),
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        if False :
            print("encoder_outputs.past_key_values[0][0].shape:" + str(encoder_outputs.past_key_values[0][0].shape))
        return BaseModelOutputWithPast(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """GIT Model with a `language modeling` head on top for autoregressive language modeling.""", GIT_START_DOCSTRING
)
class GitForCausalLM(GitPreTrainedModel):
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.git = GitModel(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.weights = None
        self.vocab = None
        self.lidar_encoder = GitLidarModel(config)
        self.text_projection = lambda x:x 
        self.vision_projection = lambda x:x ## identity
        self.temp = 0.07 * torch.ones([])
        # Initialize weights and apply final processing
        self.post_init()
        self.acc = 0
        
    def set_tokenizer(self, tt,maxlen):
        return None
        

    def set_lidar_model(self,lm,sop=None,use_sop=False,root_path="/tmp/") :

        self.lidar_encoder.lidar_model = lm
        self.lidar_encoder.sop = sop
        self.lidar_encoder.root_path=root_path
        self.lidar_encoder.use_sop = use_sop
        self.git.lidar_encoder.lidar_model = lm
        self.git.lidar_encoder.sop = sop
        self.git.lidar_encoder.root_path=root_path
        self.git.lidar_encoder.use_sop = use_sop        

        
    def set_vocab(self,ww) :
        self.vocab = ww


    def set_weights(self,ww) :
        self.weights = ww


    def get_input_embeddings(self):
        return self.git.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.git.set_input_embeddings(new_embeddings)
        
    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,            
        lidar_values: Optional[dict] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        truth: Optional[torch.Tensor] = None, # ajout
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Examples:

        Image captioning example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForCausalLM
        >>> import requests
        >>> from PIL import Image

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        >>> generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> print(generated_caption)
        two cats sleeping on a pink blanket next to remotes.
        ```

        Visual question answering (VQA) example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForCausalLM
        >>> from huggingface_hub import hf_hub_download
        >>> from PIL import Image

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

        >>> file_path = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
        >>> image = Image.open(file_path).convert("RGB")

        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> question = "what does the front of the bus say at the top?"

        >>> input_ids = processor(text=question, add_special_tokens=False).input_ids
        >>> input_ids = [processor.tokenizer.cls_token_id] + input_ids
        >>> input_ids = torch.tensor(input_ids).unsqueeze(0)

        >>> generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        >>> print(processor.batch_decode(generated_ids, skip_special_tokens=True))
        ['what does the front of the bus say at the top? special']
        ```

        Video captioning example:

        ```python
        >>> import av
        >>> import numpy as np
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download
        >>> from transformers import AutoProcessor, AutoModelForCausalLM

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")

        >>> # set seed for reproducability
        >>> np.random.seed(45)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # load video
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample frames
        >>> num_frames = model.config.num_image_with_embedding
        >>> indices = sample_frame_indices(
        ...     clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
        ... )
        >>> frames = read_video_pyav(container, indices)

        >>> pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values

        >>> generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

        >>> print("Generated caption:", processor.batch_decode(generated_ids, skip_special_tokens=True))
        Generated caption: ['a woman is sitting at a table and she is talking about the food she is holding.']
        ```
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        # print("=== FORWARD IN ===")
        # print("input_ids:" + str(input_ids))
        # print("att_maskk:" + str(attention_mask))
        # print("labels   :" + str(labels))
        
        # if past_key_values is not None :
        #     print("past_key_len:" + str(len(past_key_values)))
   
        # print(lidar_values['batch_size'])
        # if points is not None :
        #     print(points.shape)
        # else :
        #     print(" PONT IS NONE")

        # print(" ")


        if labels is not None :
            attention_mask[input_ids == 102] = 0
            input_ids[input_ids == 102] = 0

        #frame_id = lidar_values['frame_id']
        # pos_id = lidar_values['id_pcd_positif']
        # neg_id = lidar_values['id_pcd_negatif']

        if self.training:
            do_use_contrast = True
            do_use_contrast_quad = True
        else:
            do_use_contrast = False
            do_use_contrast_quad = False

        #pixel_values = torch.rand(10, 3, 224, 224)
    
        outputs = self.git(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            lidar_values=lidar_values,
            input_mod='frame_id',
            points=points,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        if do_use_contrast :
            #lidar_values['frame_id'] = pos_id
            outputs_pos = self.git(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                lidar_values=lidar_values,
                input_mod='id_pcd_positif',
                points=points,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            #lidar_values['frame_id'] = neg_id
            outputs_neg = self.git(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                lidar_values=lidar_values,
                input_mod='id_pcd_negatif',
                points=points,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if do_use_contrast_quad :
                #lidar_values['frame_id'] = pos_id
                outputs_neg_bis = self.git(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    pixel_values=pixel_values,
                    lidar_values=lidar_values,
                    input_mod='other_id_pcd_negatif',
                    points=points,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        #lidar_values['frame_id'] = frame_id
            
        sequence_output = outputs[0]
        logits = self.output(sequence_output)

        loss_tot = None


        #import pdb; pdb.set_trace()            
        if labels is not None:            
            num_image_tokens = self.git.num_feat_lidar
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()

            labels = labels[:, 1:].contiguous()
            if self.weights is not None :
                weights = self.weights.to(device=shifted_logits.device,dtype=shifted_logits.dtype)
                loss_fct = CrossEntropyLoss(ignore_index=-100,weight=weights)
            else :
                loss_fct = CrossEntropyLoss(ignore_index=-100,label_smoothing=0)
            
            loss_lm = (
                loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1)) 
            )
            del shifted_logits

            if do_use_contrast : 
                sequence_outputs_pos = outputs_pos[0]
                logits_pos = self.output(sequence_outputs_pos)
                shifted_logits_pos = logits_pos[:, num_image_tokens:-1, :].contiguous()
                del logits_pos
                loss_lm_pos = (
                    loss_fct(shifted_logits_pos.view(-1, self.config.vocab_size), labels.view(-1)) #+
                    )
                del shifted_logits_pos


                sequence_output_neg = outputs_neg[0]
                logits_neg = self.output(sequence_output_neg)
                shifted_logits_neg = logits_neg[:, num_image_tokens:-1, :].contiguous()
                del logits_neg
                loss_lm_neg = (
                    loss_fct(shifted_logits_neg.view(-1, self.config.vocab_size), labels.view(-1)) #+
                    ) 
                del shifted_logits_neg
                
                
                if do_use_contrast_quad :
                    sequence_output_neg_bis = outputs_neg_bis[0]
                    logits_neg_bis = self.output(sequence_output_neg_bis)
                    shifted_logits_neg_bis = logits_neg_bis[:, num_image_tokens:-1, :].contiguous()
                    del logits_neg_bis
                    loss_lm_neg_bis = (
                        loss_fct(shifted_logits_neg_bis.view(-1, self.config.vocab_size), labels.view(-1)) #+
                    ) 
                    del shifted_logits_neg_bis
            
            if do_use_contrast : 
   
                metap=0.5
                loss_tot =  loss_lm   + (loss_lm_pos - loss_lm_neg + metap).clamp(min=0.0)
                if do_use_contrast_quad :
                    print("quad loss")
                    metbe=0.3
                    loss_tot =  loss_lm   + (loss_lm_pos - loss_lm_neg + metap).clamp(min=0.0) + (loss_lm_pos - loss_lm_neg_bis + metbe).clamp(min=0.0)
                print("----")
                print(loss_tot)
                print(loss_lm)
                print(loss_lm_pos)
                print(loss_lm_neg)                                
            else :
                loss_tot =  loss_lm

            self.acc = self.acc +1
       
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss_tot,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": kwargs.get("pixel_values", None),
            "points": kwargs.get("points", None),
            "lidar_values": kwargs.get("lidar_values", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past