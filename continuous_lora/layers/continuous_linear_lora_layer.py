from continuous_lora.layers.continuous_lora_base_layer import ContinuousLoRALayer
from  torch import nn, Tensor
import torch.nn.functional as F


class ContinuousLinearLoRALayer(nn.Linear, ContinuousLoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        number_of_tasks: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        self.parents_initialized = False
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ContinuousLoRALayer.__init__(self, d=in_features, k=out_features, r=r, scaling=r/lora_alpha, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights, number_of_tasks=number_of_tasks)
        
        assert isinstance(r, int)
        assert r > 0
        self.parents_initialized = True

        self.fan_in_fan_out = fan_in_fan_out
    
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def count_trainable_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_parameters(self) -> None:
        if self.parents_initialized:
            nn.Linear.reset_parameters(self)
            ContinuousLoRALayer.reset_parameters(self)


    def train(self, mode: bool = True):
        def T(w: Tensor) -> Tensor:
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        # nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: Tensor) -> Tensor:
        def T(w: Tensor) -> Tensor:
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
