# Domain Adaptation Project (CS229)

# Data
## CelebA Data
Create celebA dataset splits:
```python split_data.py```

View stats on each celebA dataset split:
```python get_split_info.py```

## Camelyon17 Data
Download data (do only once)
1. `pip install wilds`
2. Run: 
```
from wilds import get_dataset
dataset = get_dataset(dataset=“camelyon17”, download=True)
```

# Experiments / training
## Steps to train vanilla ResNet (CAMELYON17)
Using the debug dataset, run:
```python main.py launch_training --config_filename="config_camelyon.yaml" --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/vanilla_ResNet_debug" --train_metadata="debug/debug_metadata_train.csv" --val_metadata="debug/debug_metadata_val.csv" --test_metadata=debug/debug_metadata_test.csv```

Using the main dataset, run:
```python main.py launch_training --config_filename=config_camelyon.yaml --data_dir=data/camelyon17_v1.0 --experiment_name=experiments/camelyon/vanilla_Resnet_0609_seed2 --train_metadata=wilds_splits/metadata_train.csv --val_metadata=wilds_splits/metadata_val.csv --test_metadata=wilds_splits/metadata_test.csv```

## Steps to train DANN method (CAMELYON17)
Using the debug dataset, run:
```python dann.py launch_training --config_filename="config_camelyon.yaml" --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/DANN_debug" --train_metadata="debug/debug_metadata_train.csv" --data_unlabeled_dir=data/camelyon17_unlabeled_v1.0 --train_target_unlabeled_metadata="debug/debug_metadata_target_unlabeled.csv" --val_metadata="debug/debug_metadata_val.csv" --test_metadata=debug/debug_metadata_test.csv --test_metadata=wilds_splits/metadata_test.csv```

Using the main dataset, run:
```python dann.py launch_training --config_filename=config_camelyon.yaml --data_dir=data/camelyon17_v1.0 --experiment_name=experiments/camelyon/DANN_new_hyperparams --train_metadata=wilds_splits/metadata_train.csv --val_metadata=wilds_splits/metadata_val.csv --data_unlabeled_dir=data/camelyon17_unlabeled_v1.0 --train_target_unlabeled_metadata=unlabeled_hospital4.csv --test_metadata=wilds_splits/metadata_test.csv```

## Evaluation
To evaluate model at an epoch and save predictions to a json file:
Using the debug dataset, run:
```python main.py eval_checkpoint --config_name=config_camelyon.yaml --exp_dir=experiments/camelyon/DANN_debug --ckpt_name=epoch2_model.pth --data_dir=data/camelyon17_v1.0 --dataset_metadata=debug/test/metadata_debug_test.csv```

To evaluate model at an epoch:
```python main.py eval_checkpoint --exp_name="vanilla_ResNet_debug" --epoch=0 --dataset_metadata="debug/train/metadata_debug_train.csv"```
```python main.py eval_checkpoint --config_name=config_camelyon.yaml --exp_dir=experiments/camelyon/DANN_new_hyperparams_seed4 --ckpt_name=epoch16_model.pth --data_dir=data/camelyon17_v1.0 --dataset_metadata=wilds_splits/metadata_test.csv``

## Environment set up
Run `conda activate CS229`

Create a new tmux session when running experments. 
To set up tmux:
`conda install -c conda-forge tmux`