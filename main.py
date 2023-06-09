import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.data_loader import get_camelyon_data_loader, get_celeba_data_loader, get_fmow_data_loader
import fire
import multiprocessing
import yaml
#from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models import resnet18, resnet50
from pathlib import Path
import wandb
import os
from models.resnet50 import ResNet, Block
import shutil
import numpy as np
import json
import random

num_cpus = multiprocessing.cpu_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

def train_multiclass(model, train_loader, criterion, optimizer, config):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, metadata) in enumerate(tqdm(train_loader)):
        # intputs shape:  torch.Size([8, 3, 224, 224])
        # target shape:  torch.Size([8, 1])
        inputs, targets = inputs.to(device), targets.to(device) 
        
        optimizer.zero_grad()

        # represents the probability that the input belongs to the positive class (i.e., class 1).
        targets = targets.squeeze().long() # torch.Size([8])
        outputs = model(inputs) # torch.Size([8, 62])

        loss = criterion(outputs, targets) # float
        
        # L2 regularization
        regularization_term = 0.0
        if config["reg_lambda"] != 0:
            for param in model.parameters():
                regularization_term += torch.norm(param, 2)**2
            regularization_term *= config["reg_lambda"]
        loss += regularization_term

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        # Update training accuracy
        #predicted = (outputs > 0.5).float()
        #_, predicted = torch.max(outputs, 1) # tensor([16, 16, 16, 16, 16, 22, 16, 16])
        predicted = torch.argmax(outputs, dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calculate average training loss and accuracy
    avg_train_loss = train_loss / len(train_loader) # divide by num batches
    train_accuracy = 100. * correct / total

    return train_accuracy, avg_train_loss

def train(model, train_loader, criterion, optimizer, config):
    """
    Train for 1 epoch
    Returns the average train accuracy and average train loss (across all batches for this one epoch)
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, metadata) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device) # target shape: torch.Size([32, 1])
        
        optimizer.zero_grad()

        # represents the probability that the input belongs to the positive class (i.e., class 1).
        outputs = model(inputs) # output shape: torch.Size([32, 1])

        if config["upweighting"] == False:
            loss = criterion(outputs, targets) # float
        else:
            # Loss upweighting: in train, 20% of men. We want to upweight men to reach 80% (their representation in test distribution)
            men_mask = metadata['Male'] == 1  # torch.Size([32]) of False, True
            upweights = torch.ones((len(targets), 1)).to(device) # shape [batch_size, 1]
            upweights[men_mask] = 4 # upweight by 4x

            criterion = nn.BCELoss(reduction='none')
            loss = criterion(outputs, targets) # torch.Size([32, 1])
            loss = torch.mean(loss * upweights)
        
        # L2 regularization
        regularization_term = 0.0
        if config["reg_lambda"] != 0:
            for param in model.parameters():
                regularization_term += torch.norm(param, 2)**2
            regularization_term *= config["reg_lambda"]
        loss += regularization_term

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        # Update training accuracy
        predicted = (outputs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calculate average training loss and accuracy
    avg_train_loss = train_loss / len(train_loader) # divide by num batches
    train_accuracy = 100. * correct / total

    return train_accuracy, avg_train_loss

def test_multiclass(model, test_loader, criterion, save_to_file=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Update test accuracy
            #predicted = (outputs > 0.5).float()
            #_, predicted = torch.max(outputs, 1)
            predicted = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if save_to_file:
                for i in range(len(inputs)):
                    prediction = {
                        'image_path': metadata['image_path'][i],
                        'prediction': int(predicted[i].item())
                    }
                    predictions.append(prediction)
    
    # Calculate average training loss and accuracy
    avg_test_loss = test_loss / len(test_loader) # divide by num batches
    test_accuracy = 100. * correct / total

    if save_to_file:
        # Write predictions to JSON file
        with open(save_to_file, 'w') as json_file:
            json.dump(predictions, json_file)

    return test_accuracy, avg_test_loss

def test(model, test_loader, criterion, save_to_file=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Update test accuracy
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if save_to_file:
                for i in range(len(inputs)):
                    prediction = {
                        'image_path': metadata['image_path'][i],
                        'prediction': int(predicted[i].item())
                    }
                    predictions.append(prediction)
    
    # Calculate average training loss and accuracy
    avg_test_loss = test_loss / len(test_loader) # divide by num batches
    test_accuracy = 100. * correct / total

    if save_to_file:
        # Write predictions to JSON file
        with open(save_to_file, 'w') as json_file:
            json.dump(predictions, json_file)

    return test_accuracy, avg_test_loss


def get_model(config):
    if config["pretrained"] == False:
        # Use custom class
        model = ResNet(config["resnet_size"], Block, image_channels=3, num_classes=config["num_classes"])
        print("model: ", model)
    else:
        # Use pretrained model
        print("Using pretrained model!")
        if config["resnet_size"] == 50:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif config["resnet_size"] == 18:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            raise Exception("Invalid resnet_size in config.")
        
        # Modify the last fully connected layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, config["num_classes"]),
            nn.Sigmoid()
        )
        
        if config["freeze_layers"] == True:
            print("Freezing layers!")
            # Freeze all layers except the last one
            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True

    return model

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def launch_training(config_filename, data_dir, experiment_name, train_metadata, val_metadata, test_metadata=None, ckpt_pth=None):
    experiment_dir = Path(experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Get config / hyperparams
    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if ckpt_pth is None:
        shutil.copy(config_filename, experiment_dir / config_filename)
    else:
        pth = config_filename.split("/")
        config_last_name = pth[-1]
        shutil.copy(config_last_name, experiment_dir / config_last_name)

    # Set seeds
    set_seed(config["seed"])

    # Initialize w&b logging
    wandb.init(
        project="domain-adaptation",
        config=config
    )

    # Get data
    train_loader = get_camelyon_data_loader(data_dir, train_metadata, config)
    val_loader = get_camelyon_data_loader(data_dir, val_metadata, config)
    #train_loader = get_fmow_data_loader(data_dir, train_metadata, batch_size=config["batch_size"])
    #val_loader = get_fmow_data_loader(data_dir, val_metadata, batch_size=config["batch_size"])
    if test_metadata:
        test_loader = get_camelyon_data_loader(data_dir, test_metadata, config)
    
    # Get model
    if ckpt_pth is None:
        model = get_model(config)
    else:
        print("Resuming from checkpoint!")
        model = load_model_from_checkpoint(config_filename, ckpt_pth)
    model = model.to(device)

    # Set up loss
    if config["loss_criterion"] == "binary-cross-entropy":
        criterion = nn.BCELoss()
    elif config["loss_criterion"] == "cross-entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Need to define loss criterion")

    # Set up optimizers
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=float(config["weight_decay"]))
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_scheduler_step_size"], gamma=config["lr_scheduler_gamma"])

    train_fn, test_fn = None, None
    if config["dataset_name"] == "fmow":
        train_fn = train_multiclass
        test_fn = test_multiclass
    elif config["dataset_name"] == "CelebA" or config["dataset_name"] == "camelyon17":
        train_fn = train
        test_fn = test
    else:
        raise Exception("Invalid dataset name")

    for epoch in range(config["num_epochs"]):
        print("Epoch: ", epoch)

        # Train
        train_acc, train_loss = train_fn(model, train_loader, criterion, optimizer, config)
        print(f"Epoch: {epoch}, Train loss: {train_loss}, Train accuracy: {train_acc}")
        
        #scheduler.step()

        # Eval 
        valid_acc, valid_loss = test_fn(model, val_loader, criterion)
        print(f"Epoch: {epoch}, Validation loss: {valid_loss}, Validation accuracy: {valid_acc}")

        # Eval ID
        if test_metadata:
            test_acc, test_loss = test_fn(model, test_loader, criterion)
            print(f"Epoch: {epoch}, Validation ID loss: {test_loss}, Validation ID accuracy: {test_acc}")
        
        # https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc}, 
                    experiment_dir / f'epoch{epoch}_model.pth')

        wandb.log({"train/train_loss": train_loss,
                   "train/train_acc": train_acc,
                   "validation/valid_loss": valid_loss,
                   "validation/valid_acc": valid_acc
                   })
        if test_metadata:
            wandb.log({
                "test/test_loss": test_loss,
                "test/test_acc": test_acc
            })

def load_model_from_checkpoint(config_pth, ckpt_path):
    with open(config_pth) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        model = get_model(config)

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("train loss at checkpoint: ", checkpoint["train_loss"])
    print("train acc at checkpoint: ", checkpoint["train_acc"])
    return model

def eval_checkpoint(config_name, exp_dir, ckpt_name, data_dir, dataset_metadata):
    exp_dir = Path(exp_dir)
    config_pth = exp_dir / config_name
    ckpt_pth = exp_dir / ckpt_name
    
    with open(config_pth) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model_from_checkpoint(config_pth, ckpt_pth)
    model = model.to(device)
    
    loader = get_camelyon_data_loader(data_dir, dataset_metadata, batch_size=config["batch_size"])
    
    import datetime
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    predictions_filename = f"predictions_{date_string}.json"
    test_accuracy, avg_test_loss = test(model, loader, nn.BCELoss(), save_to_file=exp_dir / predictions_filename)
    
    print("test_accuracy: ", test_accuracy)
    print("avg_test_loss: ", avg_test_loss)

if __name__ == '__main__':
    fire.Fire()
