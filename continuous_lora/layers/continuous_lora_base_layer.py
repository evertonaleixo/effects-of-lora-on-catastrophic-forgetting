from typing import Dict, Any
import math
from  torch import nn, Tensor

class ContinuousLoRALayer():
    def __init__(
        self, 
        d: int, 
        k: int, 
        r: int, 
        lora_alpha: int, 
        scaling: float,
        lora_dropout: float,
        merge_weights: bool,
        number_of_tasks: int,
    ):
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        assert isinstance(number_of_tasks, int)
        assert number_of_tasks > 0
        self.number_of_tasks = number_of_tasks
        self.current_task = 0

        self.lora_adapters: Dict[int, Dict[str,Any]] = {
            task_id: {
                'lora_A': nn.Parameter(Tensor().new_zeros((r, d))),
                'lora_B': nn.Parameter(Tensor().new_zeros((k, r))),
                'scaling': scaling,
                'lora_alpha': lora_alpha,
                'r': r,
            }
        for task_id in range(self.number_of_tasks)}

        self.r = self.lora_adapters[self.current_task].get('r')
        self.lora_alpha = self.lora_adapters[self.current_task].get('lora_alpha')
        self.lora_A = self.lora_adapters[self.current_task].get('lora_A')
        self.lora_B = self.lora_adapters[self.current_task].get('lora_B')
        self.scaling = self.lora_adapters[self.current_task].get('scaling')
        
    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        for task_params in self.lora_adapters.values():
            nn.init.kaiming_uniform_(task_params.get('lora_A'), a=math.sqrt(5))
            nn.init.zeros_(task_params.get('lora_B'))

    def change_to_task(self, task_id:int) -> None:
        if task_id == self.current_task:
            print(f'It is already in the task {task_id}. Nothing to do.')
            return
        
        self.r = self.lora_adapters[task_id].get('r')
        self.lora_alpha = self.lora_adapters[task_id].get('lora_alpha')
        self.lora_A = self.lora_adapters[task_id].get('lora_A')
        self.lora_B = self.lora_adapters[task_id].get('lora_B')
        self.scaling = self.lora_adapters[task_id].get('scaling')

        print(f'Changing to task {task_id}')
        self.current_task = task_id

