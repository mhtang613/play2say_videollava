import torch
import torch.nn as nn
import re

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
    

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            LoraLinear(channels, channels),
            nn.GELU(),
            LoraLinear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return LoraLinear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [LoraLinear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(LoraLinear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
