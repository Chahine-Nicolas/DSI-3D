# coding=utf-8
# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch BLIP-2 model."""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch.distributed as dist
import numpy as np
import torch
import torch.utils.checkpoint
import os
import re
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig

# from pcdet.models import build_network

PRINT_ITER=10

_CHECKPOINT_FOR_DOC = "Salesforce/blip2-opt-2.7b"

BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip2-opt-2.7b",
    # See all BLIP-2 models at https://huggingface.co/models?filter=blip
]

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs/test')


def print_module_stats(ss,mm) :
    print("---- " + str(ss) + " ----")
    print(mm.shape)
    print( str(mm.std().item()) + " " + str(mm.mean().item()) + " " + str(mm.min().item()) + " " + str(mm.max().item()) )

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]    

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    # if not is_dist_avail_and_initialized():
    #     return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output    


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndCond(BaseModelOutputWithPoolingAndCrossAttentions):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    
@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.blip.modeling_blip.BlipVisionEmbeddings with Blip->Blip2
class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Blip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]

        #print("BEGIN vision forward")
        #print(patch_embeds)
        #print("  patch embed shape =>" + str(patch_embeds.shape))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        #print("  embeddings shape =>" + str(embeddings.shape))
        #print("END vision forward")

        return embeddings


class Blip2Attention(nn.Module):
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
        self.dropout = nn.Dropout(config.attention_dropout)

        # small tweak here compared to CLIP, no bias here
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


# Copied from transformers.models.blip.modeling_blip.BlipMLP
class Blip2MLP(nn.Module):
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


# Copied from transformers.models.blip.modeling_blip.BlipEncoderLayer with Blip->Blip2
class Blip2EncoderLayer(nn.Module):
    def __init__(self, config: Blip2Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Blip2Attention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Blip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Blip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Blip2Config
    base_model_prefix = "blip"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Blip2Attention", "T5Block", "OPTDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, Blip2VisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)

        elif isinstance(module,nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Blip2Encoder):
            module.gradient_checkpointing = value


BLIP_2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Blip2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BLIP_2_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for
            details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

BLIP_2_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

BLIP_2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for
            details.

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
            provided to serve as text prompt, which the language model can continue.

            Indices can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an
            encoder-decoder language model (like T5) is used.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)

        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            Only relevant in case an encoder-decoder language model (like T5) is used.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.blip.modeling_blip.BlipEncoder with Blip->Blip2
class Blip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].

    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
    """

    def __init__(self, config: Blip2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Copied from transformers.models.blip.modeling_blip.BlipVisionModel with Blip->Blip2, BLIP->BLIP_2
class Blip2VisionModel(Blip2PreTrainedModel):
    main_input_name = "pixel_values"
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Blip2VisionEmbeddings(config)
        self.encoder = Blip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        #self.post_init()

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Blip2VisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        #import pdb; pdb.set_trace()                
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


class Blip2QFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        hh = hidden_states.contiguous()
        mixed_query_layer = self.query(hh)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #import pdb; pdb.set_trace()                               
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
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
        #print("   =>" + str(attention_scores.shape))
        #print("   =>" + str(attention_mask.shape))
        #import pdb; pdb.set_trace()                               
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Blip2QFormer
class Blip2QFormerSelfOutput(nn.Module):
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


class Blip2QFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attention = Blip2QFormerMultiHeadAttention(config, is_cross_attention)
        self.output = Blip2QFormerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Blip2QFormer
class Blip2QFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states.contiguous())
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Blip2QFormer
class Blip2QFormerOutput(nn.Module):
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


class Blip2QFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(config)
        self.layer_idx = layer_idx
        self.cross_attention_frequency = config.cross_attention_frequency
        if layer_idx % self.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate =  Blip2QFormerIntermediate(config)
        self.output = Blip2QFormerOutput(config)
            
        self.intermediate_query = Blip2QFormerIntermediate(config)
        self.output_query = Blip2QFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):

        if encoder_hidden_states is None :
            self.has_cross_attention = False
        elif self.layer_idx % self.cross_attention_frequency == 0:
            self.has_cross_attention = True
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                #print(cross_attention_outputs)
                query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class Blip2QFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
        is_decoder=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

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
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    

class BertGenerationOnlyLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        logits = self.decoder(hidden_states)
        return logits

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias



# class FilteringOutput(nn.Module):
#     def __init__(self):
#     def forward(self, x):
        

class Blip2QFormerModel(Blip2PreTrainedModel):
    _tied_weights_keys = ["output.weight"]
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: Blip2QFormerConfig):
        super().__init__(config)
        self.config = config
        self.config.query_length = 32
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = Blip2QFormerEncoder(config)
        self.output_embeddings = nn.Linear(config.hidden_size, self.config.vocab_size)
        self.input_embeddings = None
        self.tokenizer = None
        self.max_txt_len = -1
        #self.post_init()
        self.vocab = None
        self.weights = None
        self.output_embeddings.weight.data.normal_(mean=1.0, std=0.01)

        self.acc = 0
    def set_weights(self,ww) :
        self.weights = ww

    def set_vocab(self,vv) :
        self.vocab = vv
        
    def get_input_embeddings(self):
        return self.embeddings.input_embedding.word_embeddings
    def set_input_embeddings(self, value,word):
        with torch.no_grad():
            if value is not None :
                self.input_embeddings = value
                self.input_embeddings.to(dtype=self.dtype)
            if word is not None :
                self.input_embeddings.word_embeddings.weight.copy_(word.weight.to(dtype=self.dtype))
    def get_output_embeddings(self):
        return self.output_embeddings 
    def set_output_embeddings(self, value):
        with torch.no_grad():
            self.output_embeddings.weight.copy_(value.weight.to(self.dtype))
            self.output_embeddings.bias.copy_(value.bias.to(self.dtype))

    def set_tokenizer(self, tt,maxlen):
        self.tokenizer = tt
        self.max_txt_len = maxlen

        
    def set_classifier(self, value):
        self.cls = value.to(dtype = self.dtype)        

    def set_bert(self, value):
        self.bert_model = value.to(dtype = self.dtype)        
        
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        is_decoder: bool,            
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:            
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:  # UniLM style attention mask
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds: torch.FloatTensor=None,
        input_ids=None,            
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids=None,            
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels=None,
        is_decoder=False,
        is_only_encoder=False,            
            
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False

        
        if input_ids is None:
            assert (
                query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None"
        
        if past_key_values is not None:
            if query_embeds is not None :
                print("ERRROR")
                query_embeds = None            

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length
            if past_key_values is not None
            else 0
        )        


        query_length = query_embeds.shape[1] if query_embeds is not None else 0
        
        embedding_output = self.input_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=query_embeds,
            past_key_values_length=past_key_values_length,
        ).contiguous().to(dtype=self.dtype)
        #query_length = embedding_output.shape[1] if embedding_output is not None else 0

        #import pdb; pdb.set_trace()                 
        
        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None :
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_ids.shape,
                device,
                is_decoder,
                has_query=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device, is_decoder)


        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )

        
        # if input_ids is not None : 
        #     print(str(input_shape) + " - " + str(past_key_values_length) + " - "  + str(embedding_output.shape) + " " + str(past_key_values_length) + " - " + str(input_ids.shape))
        # else :
        #     print(str(input_shape) + " - " + str(past_key_values_length) + " - "  + str(embedding_output.shape) + " " + str(past_key_values_length) )
        sequence_output = encoder_outputs[0]            
        # if is_only_encoder :
        #     return encoder_outputs


        print_module_stats("seqence_output:",sequence_output)

        if (query_embeds is not None) and (input_ids is not None) : 
            sequence_output = encoder_outputs[0][:, query_embeds.shape[1] :, :]
        
        pooled_output = sequence_output[:, 0, :]

        logits= self.output_embeddings(sequence_output)# outputs.logits if (return_dict)  else outputs[0]
        #prediction_scores = self.cls(sequence_output)
        #import pdb; pdb.set_trace()
        #print(self.output_embeddings.weight.data)
        if labels is not None:
            labels = labels.to(logits.device)
            #logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # prior_score = torch.zeros(shift_logits.shape, dtype=shift_logits.dtype).to(device=shift_logits.device)
            # prior_score[:,:,self.vocab] = 10
            # prior_score[mm_z] = -10
            # prior_score[:,:,101] = -100

            
            mse_loss = nn.MSELoss()
            mae_fct = nn.L1Loss()
            # Flatten the tokens

            if self.weights is not None :
                weights = self.weights.to(device=shift_logits.device,dtype=shift_logits.dtype)
                loss_fct = CrossEntropyLoss(reduction="mean",label_smoothing=0,ignore_index=-100,weight=weights)
            else : 
                loss_fct = CrossEntropyLoss(reduction="mean",label_smoothing=0,ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            #nn_loss = mae_fct(shift_logits.view(-1),  prior_score.view(-1))
            
            
 
            mm_z = torch.zeros(shift_logits.shape, dtype=torch.bool)
            mm_z[:] = True
            mm_z[:,:,self.vocab] = False
            shift_logits[mm_z] = -1000        
            max_v = torch.argmax(shift_logits,dim=2)



            if self.acc % PRINT_ITER == 0   :
                #import pdb; pdb.set_trace()             
                for ii in range(len(max_v)) :
                    print("----")
                    print(shift_labels[ii].cpu().numpy())
                    print(max_v[ii].cpu().numpy())
            self.acc = self.acc +1
            # print("----")                
            # for ii in range(len(max_v)) :
            #     print(max_v[ii].cpu().numpy())                

        #import pdb; pdb.set_trace()
        if labels is not None:
            return CausalLMOutputWithCrossAttentions(
                loss = loss,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
            )
        
        if not return_dict:
            print("change return, undersand why original is :")
            print("return (sequence_output, pooled_output) + encoder_outputs[1:]")
            return encoder_outputs

        
        return BaseModelOutputWithPoolingAndCrossAttentionsAndCond(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            logits=logits
        )




@add_start_docstrings(
    """
    BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer
    (Q-Former) and a language model.
    """,
    BLIP_2_START_DOCSTRING,
)
class Blip2Model(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.config = config
        
        self.vision_model = Blip2VisionModel(config.vision_config)
        
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        # if config.use_decoder_only_language_model:
        #     language_model = AutoModelForCausalLM.from_config(config.text_config)
        # else:
        #     language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # # Update _tied_weights_keys using the base model used.
        # if language_model._tied_weights_keys is not None:
        #     self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        #self.language_model = language_model

        # Initialize weights and apply final processing
        #self.post_init()

    # def get_input_embeddings(self):
    #     return None
    #     #return self.language_model.get_input_embeddings()

    # def set_input_embeddings(self, value):
    #     return None
    #     #self.language_model.set_input_embeddings(value)

    # def set_output_embeddings(self, new_embeddings):
    #     return None
    #     #self.language_model.set_output_embeddings(new_embeddings)

    # def get_output_embeddings(self) -> nn.Module:
    #     return None
    #     #return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    @add_start_docstrings_to_model_forward(BLIP_2_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast_Key_Values`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast_Key_Values`] that
                contains the language model logits, the past_key_values key values and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, Blip2Model

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt").to(device)
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        return text_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Blip2Model

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        >>> image_outputs = model.get_image_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return vision_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    def get_qformer_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        >>> qformer_outputs = model.get_qformer_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return query_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )


class Blip2LidarModel(nn.Module):
    main_input_name = "points"

    # Copied from transformers.models.clip.modeling_clip.CLIPVisionModel.__init__ with CLIP->Git
    def __init__(self,config):
        #self.vision_model = GitVisionTransformer(config)
        super().__init__()
        self.lidar_encoder = None
        self.sop = None
        self.root_path=None
        self.use_sop = False
        #self.hidden_size = 1408
        self.hidden_size = config.qformer_config.encoder_hidden_size
        self.patch_size = 14
        self.vox_d1 = 216
        self.vox_d2 = 248
        self.num_patches = (((self.vox_d1+self.vox_d2)/2) // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.lidar_projection = nn.Conv2d(in_channels=128,out_channels=self.hidden_size,kernel_size=(self.patch_size,self.patch_size), stride=(self.patch_size,self.patch_size))
        self.bt_norm2d = nn.BatchNorm2d(self.hidden_size, eps=1e-4, momentum=0.1)
        self.bt_norm1d = nn.BatchNorm1d(self.hidden_size, eps=1e-4, momentum=0.1)
        #self.ly_norm = nn.LayerNorm(self.hidden_size, eps=1e-4)
        #self.rl = nn.ReLU(inplace=True)

    def set_lidar_model(self,lm,sop=None,use_sop=False,root_path="/tmp/") :        
        self.lidar_encoder = lm
        self.sop = sop
        self.root_path=root_path
        self.use_sop = use_sop        
        
    def set_lidar_encoder(self,ll,lp=None,bt=None) :
        self.lidar_encoder = ll
        with torch.no_grad():
            if lp is not None : 
                self.lidar_projection = lp
            if bt is not None :
                self.bt_norm2d = bt

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def compute_sop(self,ret_dict) :
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Returns:
        ```"""
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #lidar_values['points'] = points
        ret_dict = self.lidar_encoder(lidar_values)
        xx = ret_dict['spatial_features']#.to(torch.float16)
        #import pdb; pdb.set_trace()                
        #sop = self.compute_sop(ret_dict)

        lab = "raw"
        if self.use_sop :
            lab = "sop_blip2"

        use_cache = True            
        nn_id = ret_dict['frame_id'][:,1]
        is_first_it = False

        xx = ret_dict['spatial_features']

        xx2 = xx.flatten(2) # torch.Size([32, 128, 53568])
        xx2 = xx2.permute(0, 2, 1) # torch.Size([32, 53568, 128])
        batch_sz, nfeat, dimfeat = xx2.shape
        xx2 = torch.reshape(xx2, (batch_sz, -1, nfeat, dimfeat)) #torch.Size([32, 1, 53568, 128])  
        
        xx_sop  = None
        if use_cache : 
            xx_sop = self.load_tensor(nn_id,lab)

        if xx_sop is None : 
            if self.use_sop  : 
                xx_sop = self.sop(xx2).to(dtype=xx.dtype)
            if use_cache :                 
                self.save_tensor(xx_sop,nn_id,lab)


        xx_sop_proj_0 = xx_sop[:,:121,:].reshape(batch_sz,-1,1408)
        xx_sop_proj_1 = xx_sop[:,:121,:].reshape(batch_sz,1408,-1)
        xx_proj_1 = self.lidar_projection(xx)
        xx_proj_1_norm = self.bt_norm2d(xx_proj_1)

        xx_sop_proj_0_norm = torch.nn.functional.normalize(xx_sop_proj_0,p=2, dim=-1)        
        xx_sop_proj_1_norm = self.bt_norm1d(xx_sop_proj_1)
        xx_sop_proj_2 = xx_sop_proj_1_norm.transpose(1, 2)
        xx_proj_2 = xx_proj_1_norm.flatten(2).transpose(1, 2)
        xx_cat = torch.cat((xx_proj_2,xx_sop_proj_0_norm),dim=1)
        #import pdb; pdb.set_trace()
        xx = xx_cat
        #xx = torch.nn.functional.normalize(xx_cat,p=2, dim=-1)        
        #xx = xx_proj_2

        print_module_stats("   xx3 :",xx)        

        return (ret_dict,xx.to(torch.float32))


@add_start_docstrings(
    """
    BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer
    (Q-Former) and a language model.
    """,
    BLIP_2_START_DOCSTRING,
)
class Blip2ModelQuerryLearning(Blip2Model):
    config_class = Blip2Config
    main_input_name = "input_ids"
    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.lidar_model = Blip2LidarModel(config)
        self.language_config = self.config.text_config
        #self.load_to_gpu = None
        self.tokenizer = None
        self.bert_model = None
        self.temp = 0.07 * torch.ones([])
        self.max_txt_len = 8
        self.vocab = None
        self.logger = logging.get_logger(__name__)
        self.di = 768
        self.do = 256


        self.input_embeddings = None #super().get_input_embeddings()
        
        self.output_embeddings = None #super().get_output_embeddings()
        #self.post_init()
        self.itm_head = nn.Linear(config.qformer_config.hidden_size, 2)
        # self.text_projection = lambda x:x 
        # self.vision_projection = lambda x:x ## identity
        self.text_projection = nn.Linear(self.di,self.do)
        self.vision_projection = nn.Linear(self.di,self.do)
        
        self.acc = 0
        
    def get_output_embeddings(self) :
        return self.output_embeddings
        
    def set_tokenizer(self, tt,maxlen):
        self.tokenizer = tt
        self.qformer.set_tokenizer(tt,maxlen)
        self.max_txt_len = maxlen
        
    def set_input_embeddings(self, value,word):
        self.qformer.set_input_embeddings(value,word)
    def set_output_embeddings(self, value):
        self.qformer.set_output_embeddings(value.to(dtype=self.dtype))        
        
    def set_weights(self,ww) :
        self.qformer.set_weights(ww)

    def set_vocab(self,ww) :
        self.vocab = ww
        self.qformer.set_vocab(ww)         
        
    def reset_q(self) :
        nn.init.zeros_(self.query_tokens)
        initializer_range = 0.02
        self.query_tokens.data.normal_(mean=0.0, std=initializer_range)

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        lidar_values: torch.FloatTensor,
        points = None,            
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,            
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache=None,            
        return_dict: Optional[bool] = None,
        truth: Optional[torch.Tensor] = None, # ajout
        is_decoder= False,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        ```"""

        image = pixel_values
        #vision_outputs_img = self.vision_model(pixel_values=pixel_values,output_attentions=output_attentions,output_hidden_states=output_hidden_states,return_dict=return_dict)

        # image_embeds = vision_outputs_img[0]
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #import pdb; pdb.set_trace()                
        ret_dict,embb = self.lidar_model(points,lidar_values) #points = None, lidar_model = GD-MAE = Blip2LidarModel
        print_module_stats("embb:",embb)
        
        #print("inside forward Blip2ModelQuerryLearning")
        #import pdb; pdb.set_trace()
        
        #self.logger.info("label=>" + str(lidar_values['text']))
        image_embeds = embb # = torch.Size([16, 255, 1408])

      
        image_embeds = torch.repeat_interleave(image_embeds,input_ids.size(0) // image_embeds.size(0), dim=0)
        

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
#        input_shape = query_tokens.size()[:-1]
        #self.qformer.set_bert(self.bert_model)

        if is_decoder :
 
            if past_key_values is None :
                full_attention_mask  = torch.ones(input_ids.size(), dtype=torch.long, device=image_embeds.device)
            else :
                past_len = past_key_values[0][0].shape[2]
                input_ids=input_ids[:,-1:]
                past_mask = torch.ones(past_len, dtype=torch.long, device=image_embeds.device)
                past_mask = torch.ones([attention_mask.shape[0],past_len], dtype=torch.long, device=image_embeds.device)
                full_attention_mask = torch.cat([past_mask,attention_mask[:,-1][:,None]], dim=1)
                query_tokens=None

            query_output = self.qformer(
                input_ids=input_ids,
                query_embeds=query_tokens,
                attention_mask=full_attention_mask, 
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                past_key_values=past_key_values,
                return_dict=return_dict,
                use_cache=True,
                is_decoder=True,
                is_only_encoder=False
            )

            return query_output

        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            is_only_encoder=True
        )

        
        image_feats = F.normalize(
            self.vision_projection(query_output.last_hidden_state),dim=-1
        )


            
        text_tokens_input_ids = input_ids#text_tokens.input_ids
        text_tokens_attention_mask = attention_mask#text_tokens.attention_mask
        #text_tokens_input_ids[labels == self.tokenizer.eos_token_id] = self.tokenizer.eos_token_id
        #text_tokens_attention_mask[labels == self.tokenizer.eos_token_id] = 1




        if labels is not None:
            # text_embeds = self.input_embeddings(text_tokens_input_ids)

            # text_query_tokens = text_embeds.to(dtype=self.qformer.dtype)

            text_output = self.qformer(
                input_ids=text_tokens_input_ids,
                attention_mask=text_tokens_attention_mask,
                return_dict=True,
                is_only_encoder=True
            )

            ## language projection in the same space
            text_feat = F.normalize(
                self.text_projection(text_output.last_hidden_state[:,0,:]),dim=-1
            )

        
        
            ## ===================    image text contrastive ================

            def concat_all_gather(x):
                return(x)
            def all_gather_with_grad(x):
                return(x)

            image_feats_all = concat_all_gather(image_feats)
            text_feat_all = concat_all_gather(text_feat)

            sim_q2t = torch.matmul(
                image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            sim_i2t = sim_i2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
            ).squeeze()

            ### ======>

            # text-image similarity: aggregate across all query tokens
            sim_t2i, _ = sim_t2q.max(-1)
            sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]


            #rank = dist.get_rank()
            rank = 0 ##dist.get_rank()

            bs = image.size(0)
            #self.logger.info("raank:" + str(rank) + " " + str(bs))
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
                image.device
            )


            #mv1=torch.argmax(sim_t2i,dim=0).cpu()
            #mv2=torch.argmax(sim_i2t,dim=0).cpu()

            ## Print
            #import pdb; pdb.set_trace()             
            #txt = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            #torch.std(image_embeds,dim=[1,2])
            # stats_tab=np.hstack([
            #     np.array(range(0,input_ids.size(0)))[:,None],
            #     lidar_values['text'][:,None],
            #     np.array(txt)[:,None],
            #     stt[:,None]]
            # )
            #print(stats_tab[stats_tab[:,1].argsort()])

            # id_ex = int(stats_tab[0,0])
            # print(text_feat[id_ex][:5])
            # print(image_feats[id_ex][:5,:5])
            #print(' '.join([str(i.item()) for i in mv1]))
            #print(' '.join([str(i.item()) for i in mv2]))

            #import pdb; pdb.set_trace()
            image_ids_string = lidar_values['frame_id'][:,1]
            image_ids =  torch.from_numpy(np.array([ int(re.findall(r'\d+',ee)[0]) for ee in image_ids_string])).view(-1,1).to(image.device).contiguous()            
            if True : #"image_id" in samples.keys(): #coco retrieval finetuning
                # image_ids = torch.from_numpy(np.array(lidar_values["text"]).astype(int)).view(-1,1).to(image.device).contiguous()

                image_ids =  torch.from_numpy(np.array([ int(re.findall(r'\d+',ee)[0]) for ee in image_ids_string])).view(-1,1).to(image.device).contiguous()
                image_ids_all = concat_all_gather(image_ids)
                pos_idx = torch.eq(image_ids, image_ids_all.t()).float()       
                sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
                sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
                loss_itc = (loss_t2i+loss_i2t)/2  
            else:                     
                loss_itc = (
                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                ) / 2

            ###============== Image-text Matching ===================###
            text_input_ids_world = concat_all_gather(text_tokens_input_ids)
            text_attention_mask_world = concat_all_gather(text_tokens_attention_mask)
            image_embeds_world = all_gather_with_grad(image_embeds)
            with torch.no_grad():
                if True : #"image_id" in samples.keys():
                    mask = torch.eq(image_ids, image_ids_all.t())
                    sim_t2i.masked_fill_(mask, -10000)
                    sim_i2t.masked_fill_(mask, -10000)
                else:    
                    sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                    sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_i2t = F.softmax(sim_i2t, dim=1)

            # select a negative image for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_tokens_input_ids, text_tokens_input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_tokens_attention_mask, text_tokens_attention_mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
                image.device
            )
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            image_embeds_all = torch.cat(
                [image_embeds, image_embeds_neg, image_embeds], dim=0
            )  # pos, neg, pos
            image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
                image.device
            )

            
            output_itm = self.qformer(
                input_ids=text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
                is_only_encoder=True
            )


            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
            vl_output = self.itm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(image.device)
            loss_itm = F.cross_entropy(logits, itm_labels)


            ##================= Image Captioning ========================##
            decoder_input_ids = text_tokens_input_ids.clone()
            decoder_input_ids[:, 0] =  self.tokenizer.bos_token_id

            #import pdb; pdb.set_trace()             
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
            #labels = labels.masked_fill(labels == self.tokenizer.eos_token_id, -100)

            decoder_input_ids[labels == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
            text_tokens_attention_mask[labels == self.tokenizer.eos_token_id] = 0



            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            attention_mask = torch.cat([query_atts, text_tokens_attention_mask], dim=1)

            lm_output = self.qformer(
                input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
                is_only_encoder=False
            )

            loss_lm = lm_output.loss

            loss_tot =  loss_lm  + loss_itm +  loss_itc
            if self.acc % PRINT_ITER == 0 :
                self.logger.info("loss_tot:" + "{0:0.5f}".format(float(loss_tot)) + " (lm:" + "{0:0.5f}".format(float(loss_lm)) + " itm:" + "{0:0.5f}".format(float(loss_itm)) + "  itc:" + "{0:0.5f}".format(float(loss_itc)) +")")
            self.acc = self.acc+1

            return Blip2ForConditionalGenerationModelOutput(
                loss=loss_tot,
                vision_outputs=None
            )


    def greedy(
        self,
        lidar_values,
        pixel_values,
        points=None,            
        use_nucleus_sampling=False,
        num_beams=11,
        num_return_sequences=11,            
        max_length=30,
        min_length=3,
        top_p=0.9,
        repetition_penalty=1.0,
        eos_token_id=None,
        pad_token_id=None,
        bos_token_id=None,            
        prefix_allowed_tokens_fn=None,
    ):


        image = pixel_values
        ret_dict,image_embeds = self.lidar_model(points,lidar_values)
        image_embeds = image_embeds.to(torch.float32)
        
        image_embeds = image_embeds.repeat(
            pixel_values.size(0) // image_embeds.size(0), 1, 1
        )
        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        input_ids = (
                torch.LongTensor(image.size(0), 1)
                .fill_(self.tokenizer.bos_token_id)
                .to(image.device)
        )

        past_key_values = None
        for ii in range(0,max_length) :
            
            
            if past_key_values is None :
                full_attention_mask  = torch.ones(input_ids.size(), dtype=torch.long)
            else :
                past_len = past_key_values[0][0].shape[2]
                input_ids=input_ids[:,-1:]
                past_mask = torch.ones(past_len, dtype=torch.long, device=image_embeds.device)
                past_mask = torch.ones([attention_mask.shape[0],past_len], dtype=torch.long, device=image_embeds.device)
                full_attention_mask = torch.cat([past_mask,attention_mask[:,-1][:,None]], dim=1)
                query_tokens=None


            query_outputs = self.qformer(
                intput_ids=input_ids,
                query_embeds=query_tokens,
                attention_mask=full_attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            #import pdb; pdb.set_trace()                            
            logits = query_outputs.logits[...,-1, :][:,None,:]
            mm_z = torch.zeros(logits.shape, dtype=torch.bool)
            mm_z[:] = True
            mm_z[:,:,self.vocab] = False
            logits[mm_z] = -1000
            
            res = torch.argmax(logits.to(torch.float32),dim=2)
            bb_mat = torch.logical_or(input_ids[:,-1]==102,input_ids[:,-1]==0)
            res[bb_mat] = 0
            input_ids = torch.hstack([input_ids,res])
            
            past_key_values = query_outputs.past_key_values
            txt_pred = self.tokenizer.batch_decode(res, skip_special_tokens=True)
            #print(input_ids)                    


        return input_ids

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        #print(input_ids)
        #print(attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": kwargs.get("pixel_values", None),
            "points": kwargs.get("points", None),
            "lidar_values": kwargs.get("lidar_values", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "is_decoder" : True,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

        
