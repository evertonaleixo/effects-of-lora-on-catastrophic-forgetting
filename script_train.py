import sys
import os

# Adiciona o diretório pai ao caminho de pesquisa de módulos
sys.path.append(os.path.abspath(os.path.join('..')))


from continuous_lora.models.lora_vgg19 import LoraVGG19
from datasets import get_dataset, ContinuousLearninDataset

from tqdm import tqdm

import wandb
import torchvision.models as models

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc
import random
from copy import deepcopy
import numpy as np

import argparse
import torch


def __start_wandb(project_name, experiment_number, learning_rate, weight_decay, batch_size, dataset_name, max_iters, patience, number_of_tasks):
    config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "architecture": "LoraVGG19",
        "dataset": f'{dataset_name.upper()}',
        "epochs": max_iters,
        "lr_schedule": f"ReduceLROnPlateau - Patience {patience} - Monitoring Val Accuracy",
        "description": f"Testing {dataset_name.upper()} splited into {number_of_tasks} tasks."
    }
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        name=f"lora-vgg19-lr-{learning_rate}-{number_of_tasks}tasks-lora-{experiment_number}",
        
        # track hyperparameters and run metadata
        config=config
    )

def __stop_wandb():
    wandb.finish()
    
def __fronzen_seeds():
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

def __get_model(dataset: ContinuousLearninDataset, r_conv:int=9, r_linear:int=30, adapt_last_n_conv:int=16, adapt_last_n_linear:int=3):
    base_model = models.vgg19_bn(weights="IMAGENET1K_V1")
    
    base_model.classifier[6] = nn.Linear(4096, dataset.metadata.total_number_classes)
    model = LoraVGG19(
        model=base_model,
        masks=dataset.masks,
        r_conv=r_conv,
        r_linear=r_linear,
        adapt_last_n_conv=adapt_last_n_conv,
        adapt_last_n_linear=adapt_last_n_linear,
    )
    # Retrain the output layer, because it is replaced with a randomized weights
    model.classifier[6] = nn.Linear(4096, dataset.metadata.total_number_classes)

    return model

def train(dataset_name, max_iters, patience, batch_size, learning_rate, weight_decay, conv_adapters, linear_adapters, experiment_number):
    # Load dataset
    ...
    print(f"""
          Train with parameters:
            dataset: {dataset_name}
            max_iters: {max_iters}
            patience: {patience}
            batch_size: {batch_size}
            learning_rate: {learning_rate}
            weight_decay: {weight_decay}
            conv_adapters: {conv_adapters}
            linear_adapters: {linear_adapters}
            experiment_number: {experiment_number}
    \n\n
    """)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = get_dataset(name=dataset_name)
    tasks = [i for i in range(dataset.metadata.n_split_experiences)]
    best_model = {}

    __fronzen_seeds()

    __start_wandb(
        project_name=f"cf-lora-{dataset_name}-loravgg19",
        experiment_number=experiment_number,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        dataset_name=dataset_name,
        max_iters=max_iters,
        patience=patience,
        number_of_tasks=dataset.metadata.n_split_experiences
    )

    model = __get_model(dataset=dataset)
    model_type = 'lora'

    for task in tasks:
        epoch_losses = {'train_acc': [], 'train_loss': [], 'test_acc': [], 'test_loss': []}
        model.to(device)
        print(40*'-', f'TASK_{task}', 40*'-')
    
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=4, verbose=True)
        
        early_stop = 0
        best_val = 0
        for i in range(max_iters):
            batch_gen = torch.utils.data.DataLoader(dataset.train_ds[task], 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=1,
                                                    )
            print(f'Training task {task} in epoch {i}. Batch size: {batch_size}.')
            total_loss = .0
            total_hit = 0
            for batch in tqdm(batch_gen):
                sample = batch[0].to(device)
                target = batch[1]
                target_onehot = F.one_hot(target, num_classes=dataset.metadata.total_number_classes).to(torch.float).to(device)

                y_hat = model(sample)
    
                # Compute the loss
                loss_training = criterion(y_hat, target_onehot)            
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step()
                
                total_loss += loss_training
                total_hit += sum(np.argmax(y_hat.cpu().detach().numpy(), axis=1) == target.numpy())

                sample.to('cpu')
                target_onehot.to('cpu')
                del sample, target_onehot
                gc.collect()
                torch.cuda.empty_cache()
                
            # Evaluate in test DS after each epoch
            with torch.no_grad():
                batch_gen_test = torch.utils.data.DataLoader(dataset.test_ds[task], 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=1,
                                                    )
                model.eval()
                test_loss = 0.
                acc = 0.
                for batch_test in tqdm(batch_gen_test):
                    sample = batch_test[0].to(device)
                    target = batch_test[1]
                    target_onehot = F.one_hot(target, num_classes=dataset.metadata.total_number_classes).to(torch.float).to(device)
        
                    y_hat = model(sample)
                    cpu_inference = y_hat.argmax(axis=1)
                    
                    test_loss += criterion(y_hat, target_onehot)
                    acc += sum(cpu_inference.cpu() == target).item()

                    sample.to('cpu')
                    target_onehot.to('cpu')
                    del sample, target_onehot
                    gc.collect()
                    torch.cuda.empty_cache()

                test_loss = (test_loss/len(batch_gen_test)).cpu()
                acc = acc/len(dataset.test_ds[task])
                lr_scheduler.step(acc)
                model.train()
            
            epoch_losses['train_acc'].append(total_hit/(len(batch_gen)*batch_size))
            epoch_losses['train_loss'].append((total_loss/len(batch_gen)).cpu().item())
            
            epoch_losses['test_acc'].append(acc)
            epoch_losses['test_loss'].append(test_loss.item())
    
            # if acc > (best_val+0.01):
            if acc > (best_val):
                model.to('cpu')
                best_model[f'{model_type}-{task}'] = deepcopy(model)
                model.to(device)
                best_val = acc
                early_stop = 0
    
            if early_stop > patience:
                break
            
            early_stop += 1

            wandb.log({"train_acc": (total_hit/(len(batch_gen)*batch_size)), "train_loss": (total_loss/len(batch_gen)), 
                        "val_acc": acc, "val_loss": test_loss, "task": (task+1),
                        "Accuracy": acc, "Loss": test_loss, 
                        "patience": early_stop, "best_val_acc": best_val, "learning_rate": optimizer.param_groups[0]['lr']
                        })
            
            print(f'Trainig acc: {total_hit/(len(batch_gen)*batch_size):.4}   //   Training loss: {(total_loss/len(batch_gen)):.4f}   //   Test acc: {acc:.4f}   //   Test loss: {test_loss:.4f}')
            print(f'early_stop: {early_stop}  /   Best acc: {best_val}')
            del batch_gen, batch_gen_test
            gc.collect()
            torch.cuda.empty_cache()

        model.to('cpu')
        del model
        model = best_model[f'{model_type}-{task}']
        del best_model[f'{model_type}-{task}']

        gc.collect()
        torch.cuda.empty_cache()
        
        # del criterion, optimizer, lr_scheduler 
        gc.collect()
        torch.cuda.empty_cache()
        target_task = task + 1
        if target_task < len(tasks): 
            model.change_to_task(target_task)

    __stop_wandb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for the optimizer.")
    parser.add_argument("--conv_adapters", type=int, default=3, help="Number of convolutional adapters.")
    parser.add_argument("--linear_adapters", type=int, default=2, help="Number of linear adapters.")
    parser.add_argument("--experiment_number", type=int, default=1, help="Number experiment to be used in logs.")

    args = parser.parse_args()

    train(
        args.dataset,
        args.max_iters,
        args.patience,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.conv_adapters,
        args.linear_adapters,
        args.experiment_number
    )
