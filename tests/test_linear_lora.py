from continuous_lora.layers.continuous_linear_lora_layer import ContinuousLinearLoRALayer
import torch

def test_create_linear_lora_layer():
    layer = ContinuousLinearLoRALayer(
        in_features=2,
        out_features=10,
        number_of_tasks=2,
        r=1,
        lora_alpha=1,
        lora_dropout=.0,
        fan_in_fan_out=False,
        merge_weights=False,
    )

    assert layer is not None

def test_create_linear_layer_ten_tasks_it_should_has_ten_lora_params():
    layer = ContinuousLinearLoRALayer(
        in_features=2,
        out_features=10,
        number_of_tasks=10,
        r=1,
        lora_alpha=1,
        lora_dropout=.0,
        fan_in_fan_out=False,
        merge_weights=False,
    )

    assert len(layer.lora_adapters.keys()) == 10

def test_create_linear_layer_two_tasks_it_should_has_all_lora_parameters():
    layer = ContinuousLinearLoRALayer(
        in_features=2,
        out_features=10,
        number_of_tasks=2,
        r=1,
        lora_alpha=1,
        lora_dropout=.0,
        fan_in_fan_out=False,
        merge_weights=False,
    )


    assert layer.lora_adapters[0].get('lora_A') is not None
    assert layer.lora_adapters[0].get('lora_B') is not None
    assert layer.lora_adapters[0].get('scaling') is not None
    assert layer.lora_adapters[0].get('lora_alpha') is not None
    assert layer.lora_adapters[0].get('r') is not None


def test_create_linear_layer_two_tasks_it_should_has_lora_param_A_r_times_kernels_per_in_channels_times_kernel_size():
    layer = ContinuousLinearLoRALayer(
        in_features=2,
        out_features=10,
        number_of_tasks=2,
        r=1,
        lora_alpha=1,
        lora_dropout=.0,
        fan_in_fan_out=False,
        merge_weights=False,
    )

    assert layer.lora_adapters[0].get('lora_A').shape == (1,2)

def test_create_linear_layer_two_tasks_it_should_has_lora_param_B_out_channels_div_times_kernel_size_per_r_times_kernels():
    layer = ContinuousLinearLoRALayer(
        in_features=2,
        out_features=10,
        number_of_tasks=2,
        r=1,
        lora_alpha=1,
        lora_dropout=.0,
        fan_in_fan_out=False,
        merge_weights=False,
    )

    assert layer.lora_adapters[0].get('lora_B').shape == (10,1)

def test_create_linear_layer_two_tasks_the_lora_multply_should_match_with_model_parameters():
    layer = ContinuousLinearLoRALayer(
        in_features=2,
        out_features=10,
        number_of_tasks=2,
        r=1,
        lora_alpha=1,
        lora_dropout=.0,
        fan_in_fan_out=False,
        merge_weights=False,
    )

    for task_id in range(2):
        ephemeral_parameters = layer.lora_adapters[task_id].get('lora_B')@layer.lora_adapters[task_id].get('lora_A')
        assert  layer.weight.shape == ephemeral_parameters.shape


def test_create_linear_layer_two_tasks_and_change_task_has_to_preserve_parameters_values():
    layer = ContinuousLinearLoRALayer(
        in_features=2,
        out_features=10,
        number_of_tasks=2,
        r=1,
        lora_alpha=1,
        lora_dropout=.0,
        fan_in_fan_out=False,
        merge_weights=False,
    )

    assert layer.current_task == 0

    lora_A_task_0 = layer.lora_A
    lora_A_task_0 = lora_A_task_0.detach()
    lora_A_task_0[0][0] = torch.tensor(1.0, requires_grad=True)
    print(lora_A_task_0)
    layer.change_to_task(1)
    assert layer.current_task == 1
    lora_A_task_1 = layer.lora_A
    lora_A_task_1 = lora_A_task_1.detach()
    lora_A_task_1[0][0] = torch.tensor(2.0, requires_grad=True)
    print(lora_A_task_1)
    layer.change_to_task(0)
    lora_A_task_0 = layer.lora_A
    assert lora_A_task_0[0][0].item() == 1.0
