import torch

from torch import nn
import torch.nn.functional as F


class Conv2dLora(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super(Conv2dLora, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    
    self.a = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, 1))
    self.b = nn.Parameter(torch.randn(1, self.in_channels, 1, self.kernel_size))
    self.c = nn.Parameter(torch.zeros(self.out_channels, self.in_channels, 1, 1))
    self.d = nn.Parameter(torch.randn(self.out_channels))
    self.d_ = nn.Parameter(torch.zeros(1))

    self.op = F.conv2d

    self.first_task_trained = False
    self.selected_task = 0
    self.task_parameters = {}

        
  def count_trainable_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
  def forward(self, x):
    if self.selected_task == 0:
    #   x = self.op(
    #       x,
    #       self.weight,
    #       self.bias,
    #       stride=self.stride,
    #       padding=self.padding,
    #   )

      ephemeral_kernels = (self.a*self.b*self.c).view(self.weight.shape)
      ephemeral_bias = self.d * self.d_
      x = self.op(
          x,
          self.weight + ephemeral_kernels ,
          self.bias + ephemeral_bias,
          stride=self.stride,
          padding=self.padding,
      )
    else:
      ephemeral_kernels = (self.a*self.b*self.c).view(self.weight.shape)
      ephemeral_bias = self.d * self.d_
      x = self.op(
          x,
          self.weight + ephemeral_kernels ,
          self.bias + ephemeral_bias,
          stride=self.stride,
          padding=self.padding,
      )
      
    return x