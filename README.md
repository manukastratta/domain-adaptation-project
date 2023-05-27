# test-time-training-project

## Data (CelebA)
Create celebA dataset splits:
```python split_data.py```

View stats on each celebA dataset split:
```python get_split_info.py```

## Steps to train vanilla ResNet (CelebA)
Using the debug dataset, run:
```python main.py launch_training --config_filename="config_celebA.yaml" --data_dir="data/CelebA" --experiment_name="experiments/CelebA/vanilla_ResNet_debug" --train_metadata="splits/debug_setting1_train.csv" --val_metadata="splits/debug_setting1_val.csv"```

Using the main dataset, run:
```python main.py launch_training --config_filename="config_celebA.yaml" --data_dir="data/CelebA" --experiment_name="experiments/CelebA/vanilla_ResNet_scratch_18" --train_metadata="splits/setting1_train.csv" --val_metadata="splits/setting1_val.csv"```

To resume training from checkpoint, add extra ckpt_pth and change config_filename to be the (relative) path to the config you want to restart from:
```python main.py launch_training --config_filename="experiments/CelebA/vanilla_ResNet_debug/config_celebA.yaml" --data_dir="data/CelebA" --experiment_name="experiments/CelebA/vanilla_ResNet_debug_resumedEpoch5" --train_metadata="splits/debug_setting1_train.csv" --val_metadata="splits/debug_setting1_val.csv" --ckpt_pth="experiments/CelebA/vanilla_ResNet_debug/epoch5_model.pth"```

## Steps to train vanilla ResNet (CAMELYON17)
Using the debug dataset, run:
```python main.py launch_training --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/vanilla_ResNet_debug" --train_metadata="debug/train/metadata_debug_train.csv" --val_metadata="debug/train/metadata_debug_val.csv"```

Using the main dataset, run:
```python main.py launch_training --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/vanilla_ResNet" --train_metadata="train/metadata_train_split.csv" --val_metadata="train/metadata_val_split.csv"```

Using the main dataset (new splits), run:
```python main.py launch_training --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/vanilla_ResNet_wildsSplits" --train_metadata="wilds_splits/metadata_train.csv" --val_metadata="wilds_splits/metadata_valid.csv"```

## Evaluation
To evaluate model at an epoch and save predictions to a json file:
[Debug example]
```python main.py eval_checkpoint --config_pth="experiments/CelebA/vanilla_ResNet_debug/config_celebA.yaml" --exp_name="CelebA/vanilla_ResNet_debug" --ckpt_path="experiments/CelebA/vanilla_ResNet_debug/epoch5_model.pth" --data_dir="data/CelebA" --dataset_metadata="splits/debug_setting1_train.csv"```

To evaluate model at an epoch:
```python main.py eval_checkpoint --exp_name="vanilla_ResNet_debug" --epoch=0 --dataset_metadata="debug/train/metadata_debug_train.csv"```

## Environment set up
Run `conda activate CS329D`

Create a new tmux session when running experments. 
To set up tmux:
`conda install -c conda-forge tmux`

## Download data (do only once)
1. `pip install wilds`
2. 
```
from wilds import get_dataset
dataset = get_dataset(dataset=“camelyon17”, download=True)
```
