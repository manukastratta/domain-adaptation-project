# test-time-training-project

## Data (CelebA)
Create celebA dataset splits:
```python split_data.py```

View stats on each celebA dataset split:
```python get_split_info.py```

## Steps to train vanilla ResNet (FMoW)
Using the debug dataset, run:
```python main.py launch_training --config_filename="config_fmow.yaml" --data_dir="data/fmow_v1.1/debug" --experiment_name="experiments/fmow/vanilla_ResNet_debug" --train_metadata="debug_train_micro.csv" --val_metadata="debug_train_micro.csv" --val_id_metadata="debug_train_micro.csv"```

Using the main dataset, run:
```python main.py launch_training --config_filename="config_fmow.yaml" --data_dir="data/fmow_v1.1" --experiment_name="experiments/fmow/vanilla_ResNet" --train_metadata="train.csv" --val_metadata="val.csv" --val_id_metadata="id/id_val.csv"```

# Steps to train DANN method (Camelyon)
Using the debug dataset, run:
```python dann.py launch_training --config_filename="config_camelyon.yaml" --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/DANN_debug" --train_metadata="debug/train/metadata_debug_train.csv" --train_target_unlabeled_metadata="debug/train/temp_metadata_debug_target_train.csv" --val_metadata="debug/train/metadata_debug_val.csv"```

[temp] Using the main dataset, run:
```python dann.py launch_training --config_filename="config_camelyon.yaml" --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/camelyon/DANN_tradeoff1" --train_metadata="wilds_splits/metadata_train.csv" --train_target_unlabeled_metadata="wilds_splits/temp_metadata_target_unlabeled.csv" --val_metadata="wilds_splits/metadata_val.csv"```

Using the main dataset, run:
```python dann.py launch_training --config_filename="config_camelyon.yaml" --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/camelyon/DANN" --train_metadata="wilds_splits/metadata_train.csv" --val_metadata="wilds_splits/metadata_val.csv" --data_unlabeled_dir="data/camelyon17_unlabeled_v1.0" --train_target_unlabeled_metadata="unlabeled_hospital4.csv"```

## Steps to train vanilla ResNet (CAMELYON17)
Using the debug dataset, run:
```python main.py launch_training --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/vanilla_ResNet_debug" --train_metadata="debug/train/metadata_debug_train.csv" --val_metadata="debug/train/metadata_debug_val.csv"```

Using the main dataset (new splits), run:
```python main.py launch_training --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/vanilla_ResNet_wildsSplits" --train_metadata="wilds_splits/metadata_train.csv" --val_metadata="wilds_splits/metadata_valid.csv"```

Using the main dataset (old splits), run:
```python main.py launch_training --data_dir="data/camelyon17_v1.0" --experiment_name="experiments/vanilla_ResNet" --train_metadata="train/metadata_train_split.csv" --val_metadata="train/metadata_val_split.csv"```


## Evaluation
To evaluate model at an epoch and save predictions to a json file:
[Debug example]
```python main.py eval_checkpoint --config_name=config_camelyon.yaml --exp_dir=experiments/camelyon/DANN_debug --ckpt_name=epoch2_model.pth --data_dir=data/camelyon17_v1.0 --dataset_metadata=debug/test/metadata_debug_test.csv```

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
