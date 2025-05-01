#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import math

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .mpt.modeling_mpt import MPTConfig, MPTForCausalLM, MPTModel
from videollava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        
        #TODO: Initialize the inherited class, nn.linear 
        super().__init__(in_features, out_features, bias, device, dtype)

        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)

            self.lora_scaling = lora_alpha/lora_rank

            #TODO: Fill in the "..."
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank, device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            #TODO: Initialize both lora_A and lora_B with torch.nn.init. Refer to the paper to see how each is initialize
            #Hint: lora_A is initialized using kaiming_uniform_ using negative slope (a) as math.sqrt(5)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #TODO: return input after the forward pass
        #TODO: Remember to use dropout on the input before multiplying with lora_B and lora_A if the weights are not merged
        W0_x = super().forward(input)

        if self.is_lora() and not self.has_weights_merged:
          BA = self.lora_B @ self.lora_A  
          W0_x += self.lora_scaling * (self.lora_dropout(input) @ BA.T)
        return W0_x

    def train(self, mode: bool = True) -> "LoRALinear":
        #TODO: Set the linear layer into train mode
        #Hint: Make sure to demerge LORA matrices if already merged
        super().train(mode)
        if self.is_lora() and self.has_weights_merged:
            BA = self.lora_B @ self.lora_A
            self.weight.data -= self.lora_scaling * BA
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        #TODO: Set the linear layer into eval mode
        #Hint: Make sure to merge LORA matrices if already demerged
        super().eval()
        if self.is_lora() and not self.has_weights_merged:
            BA = self.lora_B @ self.lora_A
            self.weight.data += self.lora_scaling*BA
            self.has_weights_merged = True
        return self
    
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    #TODO: Loop through parameters and mark some as trainable. Which ones should these be?
    #Hint: How do you mark a parameter as trainable (or not trainable)?
    for name, param in model.named_parameters():
        if "lora_" in name: 
          param.requires_grad = True
        else:
          param.requires_grad = False
    return model

class LlavaMPTConfig(MPTConfig):
    model_type = "llava_mpt"


class LlavaMPTModel(LlavaMetaModel, MPTModel):
    config_class = LlavaMPTConfig

    def __init__(self, config: MPTConfig):
        config.hidden_size = config.d_model
        super(LlavaMPTModel, self).__init__(config)

    def embed_tokens(self, x):
        return self.wte(x)


class LlavaMPTForCausalLM(MPTForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMPTConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MPTForCausalLM, self).__init__(config)

        if not config.tie_word_embeddings:
            raise ValueError('MPTForCausalLM only supports tied word embeddings')
        self.transformer = LlavaMPTModel(config)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaMPTModel):
            module.gradient_checkpointing = value

    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, images=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        input_ids, _, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, None, attention_mask, past_key_values, labels, images)
        outputs = self.transformer(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache)
        # FIXME: this is a hack to fix the multiple gpu inference issue in https://github.com/haotian-liu/LLaVA/issues/338
        logits = LoraLinear(outputs.last_hidden_state.to(self.transformer.wte.weight.device), self.transformer.wte.weight)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds is not implemented for MPT yet')
        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError('MPT does not support generation with right padding.')
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get('use_cache') == False:
                raise NotImplementedError('MPT with prefix_lm=True does not support use_cache=False.')
        else:
            prefix_mask = None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'prefix_mask': prefix_mask, 'sequence_id': sequence_id, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache', True), "images": kwargs.get("images", None)}


AutoConfig.register("llava_mpt", LlavaMPTConfig)
AutoModelForCausalLM.register(LlavaMPTConfig, LlavaMPTForCausalLM)
