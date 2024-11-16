
from continuous_lora.layers.continuous_lora_base_layer import ContinuousLoRALayer
from  torch import nn, Tensor


class ContinuousConvLoRALayer(nn.Module, ContinuousLoRALayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            number_of_tasks: int,
            conv_module: nn.Conv2d = nn.Conv2d,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float=0.,
            merge_weights: bool = True,
            **kwargs
        ):
        self.parents_initialized = False
        super(ContinuousConvLoRALayer, self).__init__()
        
        self.conv: nn.Conv2d = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        r_ = r * kernel_size
        d_ = in_channels * kernel_size
        k_ = out_channels//self.conv.groups*kernel_size

        ContinuousLoRALayer.__init__(self, d=d_, k=k_, r=r_, scaling=r/lora_alpha, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, number_of_tasks=number_of_tasks)
        assert isinstance(kernel_size, int)
        assert isinstance(r, int)
        assert r > 0
        self.parents_initialized = True


        # Freezing the pre-trained weight matrix
        self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def count_trainable_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
    def reset_parameters(self):
        if self.parents_initialized:
            ContinuousLoRALayer.reset_parameters(self)
            self.conv.reset_parameters()
        
    def train(self, mode=True):
        # super(ContinuousConvLoRALayer, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: Tensor) -> Tensor:
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)
