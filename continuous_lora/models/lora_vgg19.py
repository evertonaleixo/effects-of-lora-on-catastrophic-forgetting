from typing import List
from copy import deepcopy

import torch.nn as nn
from torch import nn, flatten, Tensor
from torchvision.models.vgg import VGG

from continuous_lora.models.continuous_lora_base_model import ContinuousLoraBaseModel
from continuous_lora.layers.continuous_linear_lora_layer import ContinuousLinearLoRALayer
from continuous_lora.layers.continuous_conv_lora_layer import ContinuousConvLoRALayer
from continuous_lora.layers.old_conv_lora_layer import Conv2dLora


class LoraVGG19(ContinuousLoraBaseModel):
    """
    A VGG19 model that implements the ContinuousLoraBaseModel interface.
    This class extends VGG19 by allowing specific parameters to be frozen 
    and adapts to different tasks via task-specific changes.
    """

    def __init__(self, model: VGG, masks: List[List[int]], r_conv:int = 1, r_linear:int = 1, lora_alpha:float = 1.0, adapt_last_n_linear:int = 0, adapt_last_n_conv:int = 0):
        super().__init__(masks=masks, r=r_linear, lora_alpha=lora_alpha)
        self.r_linear = r_linear
        self.r_conv = r_conv
        self.features: nn.Sequential = nn.Sequential()
        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier: nn.Sequential = nn.Sequential()
        
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)
    
        total_conv_layers = 0
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                total_conv_layers += 1

        total_linear_layers = 0
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                total_linear_layers += 1

        skip_change_conv_layer = total_conv_layers - adapt_last_n_conv
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                if skip_change_conv_layer == 0:
                    new_conv = ContinuousConvLoRALayer(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size[0],
                        number_of_tasks=self.number_of_tasks,
                        conv_module=nn.Conv2d,
                        r=self.r_conv,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=0,
                        merge_weights=False,
                    )
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                    new_conv.conv = layer

                    # new_conv = Conv2dLora(
                    #     in_channels=layer.in_channels, 
                    #     out_channels=layer.out_channels, 
                    #     kernel_size=layer.kernel_size[0], 
                    #     stride=layer.stride, 
                    #     padding=layer.padding
                    # )
                    # new_conv.weight.data.copy_(layer.weight.data)
                    # new_conv.bias.data.copy_(layer.bias.data)
                    # new_conv.weight.requires_grad = False
                    # new_conv.bias.requires_grad = False

                    self.features.append(new_conv)
                else:
                    skip_change_conv_layer -= 1
                    layer.requires_grad_(False)
                    self.features.append(layer)
            else:
                layer.requires_grad_(False)
                self.features.append(layer)

        skip_change_linear_layer = total_linear_layers - adapt_last_n_linear
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                if skip_change_linear_layer == 0:
                    new_linear = ContinuousLinearLoRALayer(
                        in_features=layer.in_features,
                        out_features=layer.out_features,
                        number_of_tasks=self.number_of_classes,
                        r=self.r_linear,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=.0,
                        fan_in_fan_out=False,
                        merge_weights=False
                    )
                    new_linear.weight.data.copy_(layer.weight.data)
                    new_linear.bias.data.copy_(layer.bias.data)
                    new_linear.weight.requires_grad = False
                    new_linear.bias.requires_grad = False
                    self.classifier.append(new_linear)
                else:
                    skip_change_linear_layer -= 1
                    layer.requires_grad_(False)
                    self.classifier.append(layer)
            else:
                # layer.requires_grad_(False)
                self.classifier.append(layer)


    def count_features_trainable_params(self) -> int:
        count = 0
        for l in self.features:
            if isinstance(l, ContinuousConvLoRALayer) or isinstance(l, Conv2dLora):
                count += l.count_trainable_parameters()
            elif isinstance(l, nn.Conv2d):
                count += sum(p.numel() for p in l.parameters() if p.requires_grad)

        return count
    def count_classifier_trainable_params(self) -> int:
        count = 0
        for l in self.classifier:
            if isinstance(l, ContinuousLinearLoRALayer):
                count += l.count_trainable_parameters()
            elif isinstance(l, nn.Linear):
                count += sum(p.numel() for p in l.parameters() if p.requires_grad)

        return count
    
    def count_trainable_params(self) -> int:
        return self.count_features_trainable_params() + self.count_classifier_trainable_params()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the model.
        :param x: Input tensor
        :return: Output tensor after passing through the VGG19 model
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        out = self.classifier(x)

        # Mask code
        out = self.softmax(out).clone()  
        out[:, self.masks[self.current_task]] = .0
        
        return out
  
    def freeze_main_params(self) -> None:
        """
        Freeze the main parameters (convolutional layers) of the VGG19 model.
        This prevents updates to these layers during training.
        """
        print('freeze_main_params')


    def change_to_task(self, task_id) -> None:
        """
        Adapt the model to a new task by changing the classifier.
        
        :param task_id: A task identifier to switch the model to.
        This could, for example, modify the classifier for task-specific classes.
        """
        super().change_to_task(task_id)

        new_features:nn.Sequential =  nn.Sequential()
        new_classifier: nn.Sequential = nn.Sequential()
        fixed_part_of_target_task = self.fixed_save_parts.get(self.current_task, dict())
        fixed_part_of_current_task = self.fixed_save_parts.get(self.current_task, dict())
        fixed_part_of_current_task['features'] = []
        fixed_part_of_current_task['classifier'] = []
        # Save current layers from features that are not continuous 
        for layer in self.features:
            if not isinstance(layer, ContinuousConvLoRALayer):
                fixed_part_of_current_task['features'].append(deepcopy(layer))
        # Change to the target task
        target_non_continuous_part_of_features = fixed_part_of_target_task.get('features', [])[::-1]
        for layer in self.features:
            layer_target = layer
            if isinstance(layer, ContinuousConvLoRALayer):
                layer.change_to_task(task_id=task_id)
            else:
                if len(target_non_continuous_part_of_features) > 0:
                    layer_target =  target_non_continuous_part_of_features.pop()
                else:
                    layer_target = deepcopy(layer)
            new_features.append(layer_target)

        # Save current layers from classifier that are not continuous 
        for layer in self.classifier:
            if not isinstance(layer, ContinuousLinearLoRALayer):
                fixed_part_of_current_task['classifier'].append(deepcopy(layer))
        # Change to the target task
        target_non_continuous_part_of_classifier = fixed_part_of_target_task.get('classifier', [])[::-1]
        for layer in self.classifier:
            layer_target = layer
            if isinstance(layer, ContinuousLinearLoRALayer):
                layer.change_to_task(task_id=task_id)
            else:
                if len(target_non_continuous_part_of_classifier) > 0:
                    layer_target =  target_non_continuous_part_of_classifier.pop()
                else:
                    layer_target = deepcopy(layer)
            new_classifier.append(layer_target)

        self.features = new_features
        self.classifier = new_classifier
        self.current_task = task_id
        print(f'Change to task {task_id}. Remember to set the new weights into to optmizier.')