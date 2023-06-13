import argparse
import numpy as np
import pandas as pd
import json
import sklearn.metrics
import fire
from main import eval_checkpoint

def pipeline_eval():
    exps = [
        "experiments/camelyon/target_aug_LHM_test_labeled",
        "experiments/camelyon/target_aug_LHM_test_labeled_seed2",
        "experiments/camelyon/target_aug_LHM_test_labeled_seed3"
    ]
    ckpts = [20, 20, 18] 
    ckpts = [c-1 for c in ckpts] # 0-indexed

    for i in range(len(exps)):
        exp_dir = exps[i]
        ckpt = ckpts[i]
        eval_checkpoint("config_camelyon.yaml", exp_dir, f"epoch{ckpt}_model.pth", "data/camelyon17_v1.0", "wilds_splits/metadata_test.csv")

    for exp in exps:
        print("\n\n\n\n RESULTS FOR EXP: ", exp)
        get_metrics(f"{exp}/predictions.json", "data/camelyon17_v1.0/wilds_splits/metadata_test.csv")



def get_metrics(predictions, data_csv):
    """
    Example use:
    python eval_model.py get_metrics --predictions=experiments/camelyon/DANN_new_hyperparams_seed4/predictions_2023-06-12_10:58:23.json --data_csv=data/camelyon17_v1.0/wilds_splits/metadata_test.csv
    """
    pred_df = pd.read_json(predictions)
    # img_filename  prediction for pred_df
    print('sum', np.sum(pred_df['prediction']))
    
    labels_df = pd.read_csv(data_csv)
    # img_filename,Black_Hair for labels_df
    assert len(pred_df) == len(labels_df)
    
    full_df = pred_df.merge(labels_df, how='inner', on=['image_path'])
    assert len(full_df) == len(labels_df)
    
    full_confusion_matrix = sklearn.metrics.confusion_matrix(full_df['tumor'], full_df['prediction'])
    print(full_confusion_matrix)

    # Calculate precision and recall
    # TP = full_confusion_matrix[1, 1]
    # FP = full_confusion_matrix[0, 1]
    # FN = full_confusion_matrix[1, 0]
    # TN = full_confusion_matrix[0, 0]
    TN, FP, FN, TP = full_confusion_matrix.ravel()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy_overall = (TP + TN) / (TP + FP + FN + TN)

    print("Precision:", round(precision*100, 2))
    print("Recall:", round(recall*100, 2))
    print("TN, FP, FN, TP: ", TN, FP, FN, TP)
    print("Overall Accuracy: ", round(accuracy_overall*100, 2))


if __name__ == "__main__":
    fire.Fire()
