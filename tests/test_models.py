import pytest
import re
from continuous_lora.models.lora_vgg19 import LoraVGG19
from continuous_lora.layers.continuous_linear_lora_layer import ContinuousLinearLoRALayer
from continuous_lora.layers.continuous_conv_lora_layer import ContinuousConvLoRALayer
from torch import nn
from torchvision import models

def test_create_lora_vgg19_should_not_raise_exception():
    base_model = models.vgg19()
    model = LoraVGG19(
        model=base_model,
        masks=[[0,1,2,3,4], [5,6,7,8,9]]
    )
    assert model is not None

def test_create_lora_vgg19_with_mask_of_two_tasks_should_has_two_tasks():
    base_model = models.vgg19()
    model = LoraVGG19(
        model=base_model,
        masks=[[0,1,2,3,4], [5,6,7,8,9]]
    )
    assert model.number_of_tasks == 2

def test_create_lora_vgg19_with_mask_of_two_tasks_should_has_ten_classes():
    base_model = models.vgg19()
    model = LoraVGG19(
        model=base_model,
        masks=[[0,1,2,3,4], [5,6,7,8,9]]
    )
    assert model.number_of_classes == 10

def test_create_lora_vgg19_adapting_one_linear_two_conv():
    base_model = models.vgg19()
    model = LoraVGG19(
        model=base_model,
        masks=[[0,1,2,3,4], [5,6,7,8,9]],
        adapt_last_n_conv=2,
        adapt_last_n_linear=1
    )
    
    assert isinstance(model.classifier[0], nn.Linear)
    assert isinstance(model.classifier[3], nn.Linear)
    assert isinstance(model.classifier[6], ContinuousLinearLoRALayer)

    assert isinstance(model.features[0], nn.Conv2d)
    assert isinstance(model.features[2], nn.Conv2d)
    assert isinstance(model.features[5], nn.Conv2d)
    assert isinstance(model.features[7], nn.Conv2d)
    assert isinstance(model.features[10], nn.Conv2d)
    assert isinstance(model.features[12], nn.Conv2d)
    assert isinstance(model.features[14], nn.Conv2d)
    assert isinstance(model.features[16], nn.Conv2d)
    assert isinstance(model.features[16], nn.Conv2d)
    assert isinstance(model.features[19], nn.Conv2d)
    assert isinstance(model.features[21], nn.Conv2d)
    assert isinstance(model.features[25], nn.Conv2d)
    assert isinstance(model.features[28], nn.Conv2d)
    assert isinstance(model.features[30], nn.Conv2d)
    assert isinstance(model.features[32], ContinuousConvLoRALayer)
    assert isinstance(model.features[34], ContinuousConvLoRALayer)



def test_change_task_in_lora_vgg19_with_negative_task_should_raise_value_exception():
    base_model = models.vgg19()
    model = LoraVGG19(
        model=base_model,
        masks=[[0,1,2,3,4], [5,6,7,8,9]]
    )

    expected_message = 'Task ID must be an integer on interval of number of tasks [0,2], got -1.'
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        model.change_to_task(-1)

def test_change_task_in_lora_vgg19_with_out_of_range_task_should_raise_value_exception():
    base_model = models.vgg19()
    model = LoraVGG19(
        model=base_model,
        masks=[[0,1,2,3,4], [5,6,7,8,9]]
    )

    expected_message = 'Task ID must be an integer on interval of number of tasks [0,2], got 2.'
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        model.change_to_task(2)

def test_change_task_in_lora_vgg19_with_string_task_should_raise_value_exception():
    base_model = models.vgg19()
    model = LoraVGG19(
        model=base_model,
        masks=[[0,1,2,3,4], [5,6,7,8,9]]
    )

    expected_message = "Task ID must be an integer, got 1 of type (<class 'str'>)"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        model.change_to_task("1")
