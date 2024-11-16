from abc import ABC, abstractmethod
from typing import List, Dict
import torch.nn as nn

class ContinuousLoraBaseModel(nn.Module, ABC):
    """
    Base class for Continuous Lora models.

    This class defines the main interface that any child class must implement
    to interact with the main parameters and task changes.
    """

    def __init__(self, masks: List[List[int]], r: int = 1, lora_alpha: float = 1.0):
        super().__init__()

        self.current_task: int = 0
        self.fixed_save_parts: Dict = dict()
        self.masks: List[List[int]] = masks
        self.number_of_classes: int = sum([len(set(mask)) for mask in masks])
        self.number_of_tasks: int = len(masks)
        self.r = r
        self.lora_alpha = lora_alpha


    @abstractmethod
    def freeze_main_params(self) -> None:
        """
        Freeze the main parameters of the model.
        
        This method must be implemented by subclasses to provide specific
        functionality for freezing model parameters.
        """
        pass

    @abstractmethod
    def change_to_task(self, task_id) -> None:
        """
        Change the model to adapt to a new task.
        
        This method must be implemented by subclasses to provide functionality
        for switching the model's focus to a different task.
        
        :param task_id: An identifier for the task to switch to.
        """
         # Perform validation
        if not isinstance(task_id, int):
            raise ValueError(f"Task ID must be an integer, got {task_id} of type ({type(task_id)})")
            
        if task_id < 0 or task_id >= self.number_of_tasks:
            raise ValueError(f"Task ID must be an integer on interval of number of tasks [0,{self.number_of_tasks}], got {task_id}.")
