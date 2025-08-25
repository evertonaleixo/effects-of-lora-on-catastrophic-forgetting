from copy import deepcopy
from torch import nn, flatten
import torchvision.models as models

from typing import List


'''
input layer followed by a:

* Convolution layer (32 kernel (3,3))
* ReLU activation function

* Convolution layer (32 kernel (3,3))
* ReLU activation functio
* Dropout layer
* MaxPooling layer

* Convolution layer (64 kernel (3,3))
* ReLU activation function
* Dropout layer

* Convolution layer (64 kernel (3,3))
* ReLU activation function
* Dropout layer

* Flatten layer

* Dense layer one wit(5184 units)

* Dense layers with number of classes units with Softmax.
'''
class SenaCNN_32(nn.Module):
  def __init__(self, masks, number_of_classes=10):
    super(SenaCNN_32, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.relu1 = nn.ReLU()
    
    self.conv2 = nn.Conv2d(32, 32, 3)
    self.relu2 = nn.ReLU()
    self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.dropout2 = nn.Dropout(0.2)


    # For each new Task
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.relu3 = nn.ReLU()
    self.dropout3 = nn.Dropout(0.2)
    
    self.conv4 = nn.Conv2d(64, 64, 3)
    self.relu4 = nn.ReLU()
    self.dropout4 = nn.Dropout(0.2)

    self.flatten = nn.Flatten()
    
    self.dense5 = nn.Linear(5184, 512)
    self.output = nn.Linear(512, number_of_classes)
    self.softmax = nn.Softmax(dim=-1)

    self.dedicated_part_per_task = {}
    
    self.masks = masks
    self.task_id = 0
    self.number_of_classes = number_of_classes

        
  def forward(self, x):
    x = self.relu1(self.conv1(x))
    x = self.dropout2( self.relu2( self.conv2(x) ) )
    x = self.maxpooling1(x)

    # For each new task
    x = self.dropout3( self.relu3( self.conv3(x) ) )
    x = self.dropout4( self.relu4( self.conv4(x) ) )
    
    x = self.flatten(x)

    x = self.dense5(x)
    x = self.output(x)
    
    # Mask code
    x[:, self.masks[self.task_id]] = .0
    
    return self.softmax(x)

  def freeze_main_params(self,):
    if self.task_id != 0:
      print('Only can frozen the model in task 0')
      return
    convs = [self.conv1, self.conv2, self.conv3, self.conv4]
    
    for conv in convs:
      for p in conv.parameters():
        p.requires_grad = False
    
  def change_to_task(self, task_id):
    if task_id == self.task_id:
      print(f'The model is already configured to task {task_id}')
      return

    self.dedicated_part_per_task[self.task_id] = {
      'conv3': self.conv3,
      'relu3': self.relu3,
      'dropout3': self.dropout3,
      'conv4': self.conv4,
      'relu4': self.relu4,
      'dropout4': self.dropout4,
      'flatten': self.flatten,
      'dense5': self.dense5,
      'output': self.output
    }

    if task_id not in self.dedicated_part_per_task.keys():
      self.dedicated_part_per_task[task_id] = {
        'conv3': nn.Conv2d(32, 64, 3),
        'relu3': nn.ReLU(),
        'dropout3': nn.Dropout(0.5),
        'conv4': nn.Conv2d(64, 64, 3),
        'relu4': nn.ReLU(),
        'dropout4': nn.Dropout(0.5),
        'flatten': nn.Flatten(),
        'dense5': nn.Linear(5184, 512),
        'output': nn.Linear(512, self.number_of_classes)
      }

    self.conv3 = self.dedicated_part_per_task[task_id]['conv3']
    self.relu3 = self.dedicated_part_per_task[task_id]['relu3']
    self.dropout3 = self.dedicated_part_per_task[task_id]['dropout3']
    self.conv4 = self.dedicated_part_per_task[task_id]['conv4']
    self.relu4 = self.dedicated_part_per_task[task_id]['relu4']
    self.dropout4 = self.dedicated_part_per_task[task_id]['dropout4']
    self.flatten = self.dedicated_part_per_task[task_id]['flatten']
    self.dense5 = self.dedicated_part_per_task[task_id]['dense5']
    self.output = self.dedicated_part_per_task[task_id]['output']
    
    self.task_id = task_id



'''
input layer followed by a:

[block 1 -> layer 1-2]
* Convolution layer (64 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (64 kernel (3,3))
* BatchNorm
* ReLU activation function
* MaxPooling layer

[block 2 -> layer 3-4]
* Convolution layer (128 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (128 kernel (3,3))
* BatchNorm
* ReLU activation function
* MaxPooling layer

[block 3 -> layer 5-7]
* Convolution layer (256 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (256 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (256 kernel (3,3))
* BatchNorm
* ReLU activation function
* MaxPooling layer

[block 4 -> layer 8-10]
* Convolution layer (512 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (512 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (512 kernel (3,3))
* BatchNorm
* ReLU activation function
* MaxPooling layer

[block 5 -> layer 11-13]
* Convolution layer (512 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (512 kernel (3,3))
* BatchNorm
* ReLU activation function
* Convolution layer (512 kernel (3,3))
* BatchNorm
* ReLU activation function
* MaxPooling layer

* Flatten layer

[block 6 -> layer 14]
* Dropout layer
* Dense layer (4096 units)
* ReLU activation function

[block 7 -> layer 15]
* Dropout layer
* Dense layer (4096 units)
* ReLU activation function

[block 8 -> layer 16]
* Dense layers with number of classes units with Softmax.
'''
class SenaCNN_64(nn.Module):
  def __init__(self, masks, number_of_classes=10):
    super(SenaCNN_64, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU())
    self.layer2 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(), 
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU())
    self.layer4 = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer5 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU())
    self.layer6 = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU())
    self.layer7 = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))

    # For each new Task
    self.layer8 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer9 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer10 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer11 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer12 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer13 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.fc = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(2048, 4096),
      nn.ReLU())
    self.fc1 = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU())
    self.fc2= nn.Sequential(
      nn.Linear(4096, number_of_classes))

    self.softmax = nn.Softmax(dim=-1)

    self.dedicated_part_per_task = {}
    
    self.masks = masks
    self.task_id = 0
    self.number_of_classes = number_of_classes
    self.conv_layers = [
      self.layer1,
      self.layer2,
      self.layer3,
      self.layer4,
      self.layer5,
      self.layer6,
      self.layer7,
      self.layer8,
      self.layer9,
      self.layer10,
      self.layer11,
      self.layer12,
      self.layer13,
    ]
        
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)

    # For each new task
    out = self.layer8(out)
    out = self.layer9(out)
    out = self.layer10(out)
    out = self.layer11(out)
    out = self.layer12(out)
    out = self.layer13(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    out = self.fc1(out)
    out = self.fc2(out)
    
    # Mask code
    out[:, self.masks[self.task_id]] = .0
    
    return self.softmax(out)

  def freeze_main_params(self,):
    if self.task_id != 0:
      print('Only can frozen the model in task 0')
      return
    layers = [
      self.layer1, 
      self.layer2, 
      self.layer3, 
      self.layer4, 
      self.layer5, 
      self.layer6,
      self.layer7,
    ]
    
    for layer in layers:
      for p in layer.parameters():
        p.requires_grad = False
    
  def change_to_task(self, task_id):
    if task_id == self.task_id:
      print(f'The model is already configured to task {task_id}')
      return

    self.dedicated_part_per_task[self.task_id] = {
      'layer8': self.layer8,
      'layer9': self.layer9,
      'layer10': self.layer10,
      'layer11': self.layer11,
      'layer12': self.layer12,
      'layer13': self.layer13,
      'fc': self.fc,
      'fc1': self.fc1,
      'fc2': self.fc2,
    }

    if task_id not in self.dedicated_part_per_task.keys():
      self.dedicated_part_per_task[task_id] = {
        'layer8': nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU()),
        'layer9': nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU()),
        'layer10': nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2, stride = 2)),
        'layer11': nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU()),
        'layer12': nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU()),
        'layer13': nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2, stride = 2)),
        'fc': nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(2048, 4096),
          nn.ReLU()),
        'fc1': nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(4096, 4096),
          nn.ReLU()),
        'fc2': nn.Sequential(
          nn.Linear(4096, self.number_of_classes)),
      }

    self.layer8 = self.dedicated_part_per_task[task_id]['layer8']
    self.layer9 = self.dedicated_part_per_task[task_id]['layer9']
    self.layer10 = self.dedicated_part_per_task[task_id]['layer10']
    self.layer11 = self.dedicated_part_per_task[task_id]['layer11']
    self.layer12 = self.dedicated_part_per_task[task_id]['layer12']
    self.layer13 = self.dedicated_part_per_task[task_id]['layer13']
    self.fc = self.dedicated_part_per_task[task_id]['fc']
    self.fc1 = self.dedicated_part_per_task[task_id]['fc1']
    self.fc2 = self.dedicated_part_per_task[task_id]['fc2']
    
    self.task_id = task_id



class SenaVGG19(nn.Module):

  def __init__(self, masks, number_of_classes=10):
    super(SenaVGG19, self).__init__()
    self.features: nn.Sequential = nn.Sequential()
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.classifier = nn.Sequential()
    
    self.softmax = nn.Softmax(dim=-1)
    
    self.masks = masks
    self.task_id = 0
    self.number_of_classes = number_of_classes
    self.base_features_len = 29
    self.fixed_save_parts = dict()

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = flatten(x, 1)
    out = self.classifier(x)

    # Mask code
    out[:, self.masks[self.task_id]] = .0
    
    return self.softmax(out)
  
  def freeze_main_params(self,):
    if self.task_id != 0:
      print('[WARM] Only can frozen the model in task 0')
      return
    
    for idx, layer in enumerate(self.features):
      if idx < self.base_features_len:
        for p in layer.parameters():
          p.requires_grad = False
      else:
        break

  def change_to_task(self, task_id):
    if task_id == self.task_id:
      print(f'[WARM] The model is already prepared to task {task_id}.')
      return
    
    '''
    STORE CURRENT WEIGHTS OF self.task_id
    '''
    if self.task_id not in self.fixed_save_parts:
      self.fixed_save_parts[self.task_id] = dict({
        'features': dict(),
        'classifier': dict()
      })
    
    for idx, layer in enumerate(self.features):
      if idx >= self.base_features_len:
        self.fixed_save_parts[self.task_id]['features'][idx] = deepcopy(layer.state_dict())
    for idx, layer in enumerate(self.classifier):
      self.fixed_save_parts[self.task_id]['classifier'][idx] = deepcopy(layer.state_dict())

    '''
    LOAD WEIGHTS FROM NEW TASK
     * Check if weights of new task already exists. If not, create it
     * Then, load the fixed parts, and call change_to_task in lora layers
    '''
    if task_id not in self.fixed_save_parts:
      # Create weights to new layers
      self.fixed_save_parts[task_id] = dict({
        'features': dict(),
        'classifier': dict()
      })
      for idx, layer in enumerate(self.features):
        if idx >= self.base_features_len:
          self.fixed_save_parts[task_id]['features'][idx] = deepcopy(layer.state_dict())
      for idx, layer in enumerate(self.classifier):
        self.fixed_save_parts[task_id]['classifier'][idx] = deepcopy(layer.state_dict())

    # Changing Weights to target task
    for idx, layer in enumerate(self.features):
      if idx >= self.base_features_len:
        layer.load_state_dict(self.fixed_save_parts[task_id]['features'][idx])

    for idx, layer in enumerate(self.classifier):
      layer.load_state_dict(self.fixed_save_parts[task_id]['classifier'][idx])

    self.task_id = task_id
    print(f'[INFO] Model prepared to task {self.task_id}.')


def convert_vgg_to_sena(model: models.vgg.VGG, masks:List[List[int]], number_of_classes:int) -> SenaVGG19:
    new_model = SenaVGG19(masks, number_of_classes)
    
    for layer in model.features:
        new_model.features.append(layer)

    for layer in model.classifier:
        new_model.classifier.append(layer)
    
    return new_model