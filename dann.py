import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.data_loader import get_camelyon_data_loader
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
from models.domain_discriminator import DomainDiscriminator
from models.domain_adversarial_loss import DomainAdversarialLoss
from models.ttlib_iterators import ForeverDataIterator
from utils.ttlib_utils import accuracy, binary_accuracy, AverageMeter
import torch.nn.functional as F
import random

num_cpus = multiprocessing.cpu_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

def train(model, domain_adv, train_source_loader, train_target_iter, criterion, optimizer, config):
    """
    Train for 1 epoch
    Returns the average train accuracy and average train loss (across all batches for this one epoch)
    """
    # Train modes
    model.train()
    domain_adv.train()
    
    train_loss = 0
    transfer_losses = 0
    correct = 0
    total = 0
    n_batches = len(train_source_loader)
    domain_accs = AverageMeter('Domain Acc', ':3.1f')

    for batch_idx, (inputs, targets, metadata) in enumerate(tqdm(train_source_loader)):
        x_s, labels_s = inputs.to(device), targets.to(device)
        n_examples = len(labels_s)
        #x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]
        # truncate x_t to fit size of n_examples
        if len(x_t) != len(x_s):
            if len(x_s) > len(x_t):
                n_examples = len(x_t)
                x_s = x_s[:n_examples, :]
                labels_s = labels_s[:n_examples, :]
            elif len(x_s) < len(x_t):
                n_examples = len(x_s)
                x_t = x_t[:n_examples, :]
        assert len(x_s) == len(x_t) == len(labels_s) == n_examples
        print("n_examples: ", n_examples)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x) #torch.Size([64, 1]), torch.Size([64, 512])
        y_s, y_t = y[0:n_examples, :], y[n_examples:, :]
        f_s, f_t = f[0:n_examples, :], f[n_examples:, :]

        cls_loss = F.binary_cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t, n_examples)
        domain_acc = domain_adv.domain_discriminator_accuracy
        domain_accs.update(domain_acc.item(), x_s.size(0))

        loss = cls_loss + transfer_loss * config["dann_trade_off"]

        #cls_acc = accuracy(y_s, labels_s)[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #lr_scheduler.step()


        train_loss += loss.item()
        transfer_losses += transfer_loss.item() # tensor(0.6957, grad_fn=<MulBackward0>)

        # Update training accuracy
        predicted = (y_s > 0.5).float()
        total += len(labels_s)
        correct += (predicted == labels_s).sum().item()

    # Calculate average training loss and accuracy
    avg_train_loss = train_loss / n_batches
    train_accuracy = 100. * correct / total

    return train_accuracy, avg_train_loss, domain_accs.avg, transfer_losses

def test(model, test_loader, criterion, save_to_file=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, _ = model(inputs) # ignore features
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Update test accuracy
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if save_to_file:
                for i in range(len(inputs)):
                    prediction = {
                        'img_filename': metadata['img_filename'][i],
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
        model.return_features = True # True for DANN
        print("model: ", model)
    else:
        print("Using pretrained model!")
        if config["resnet_size"] == 50:
            model = resnet50(pretrained=True)
            #model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif config["resnet_size"] == 18:
            model = resnet18(pretrained=True)
            #model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            raise Exception("Invalid resnet_size in config.")
        
        # Modify the last fully connected layer for binary classification
        num_features = model.fc.in_features
        print("HEY num_features: ", num_features)
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

def launch_training(config_filename, data_dir, experiment_name, train_metadata, val_metadata, data_unlabeled_dir, train_target_unlabeled_metadata, ckpt_pth=None):
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
    train_source_loader = get_camelyon_data_loader(data_dir, train_metadata, batch_size=config["batch_size"])
    train_target_loader = get_camelyon_data_loader(data_unlabeled_dir, train_target_unlabeled_metadata, batch_size=config["batch_size"])
    val_loader = get_camelyon_data_loader(data_dir, val_metadata, batch_size=config["batch_size"])
    
    # Forever iterators
    #train_source_iter = ForeverDataIterator(train_source_loader, device=device)
    train_target_iter = ForeverDataIterator(train_target_loader, device=device)

    # Get model
    if ckpt_pth is None:
        model = get_model(config)
    else:
        print("Resuming from checkpoint!")
        model = load_model_from_checkpoint(config_filename, ckpt_pth)
    model = model.to(device)

    # Set up domain discriminator model
    # TODO remove hardcoded 512
    domain_discri = DomainDiscriminator(in_feature=512, hidden_size=1024).to(device)


    # Set up loss
    if config["loss_criterion"] == "binary-cross-entropy":
        criterion = nn.BCELoss()
    elif config["loss_criterion"] == "cross-entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Need to define loss criterion")
    # Set up domain disciminator loss
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # Set up optimizers
    params = list( model.parameters()) + list(domain_discri.get_parameters()) # 63
    # Exclude non-tensor params
    params = [p for p in params if isinstance(p, torch.Tensor)] # 62
    optimizer = optim.SGD(  params,
                            lr=config["learning_rate"],
                            momentum=config["momentum"],
                            weight_decay=float(config["weight_decay"]))
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_scheduler_step_size"], gamma=config["lr_scheduler_gamma"])

    train_fn, test_fn = train, test
    # if config["dataset_name"] == "fmow":
    #     train_fn = train_multiclass
    #     test_fn = test_multiclass
    # elif config["dataset_name"] == "CelebA" or config["dataset_name"] == "camelyon17":
    #     train_fn = train
    #     test_fn = test
    # else:
    #     raise Exception("Invalid dataset name")

    for epoch in range(config["num_epochs"]):
        print("Epoch: ", epoch)

        # Train
        train_acc, train_loss, domain_acc, domain_loss = train_fn(model, domain_adv, train_source_loader, train_target_iter, criterion, optimizer, config)
        print(f"Epoch: {epoch}, Train loss: {train_loss}, Train accuracy: {train_acc}, Domain loss: {domain_loss}, Domain accuracy: {domain_acc}")
        
        #scheduler.step()

        # Eval 
        valid_acc, valid_loss = test_fn(model, val_loader, criterion)
        print(f"Epoch: {epoch}, Validation loss: {valid_loss}, Validation accuracy: {valid_acc}")
        
        # https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'domain_loss': domain_loss,
                    'domain_acc': domain_acc,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc}, 
                    experiment_dir / f'epoch{epoch}_model.pth')

        wandb.log({ "train/train_loss": train_loss,
                    "train/train_acc": train_acc,
                    "train/domain_loss": domain_loss,
                    "train/domain_acc": domain_acc,
                    "validation/valid_loss": valid_loss,
                    "validation/valid_acc": valid_acc
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

def eval_checkpoint(config_pth, exp_name, ckpt_path, data_dir, dataset_metadata):
    exp_dir = Path("experiments") / exp_name
    
    with open(config_pth) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model_from_checkpoint(config_pth, ckpt_path)
    model = model.to(device)

    loader = get_camelyon_data_loader(data_dir, dataset_metadata, batch_size=config["batch_size"])

    test_accuracy, avg_test_loss = test(model, loader, nn.BCELoss(), save_to_file=exp_dir / "predictions.json")
    
    print("test_accuracy: ", test_accuracy)
    print("avg_test_loss: ", avg_test_loss)

if __name__ == '__main__':
    fire.Fire()
