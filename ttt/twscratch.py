# resnet18_layer3_gn_wd_0.1_FINAL
import json
import pandas as pd 
import numpy as np

def main():
    """ Main entry point of the app """
    results = pd.read_json('results/resnet18_layer3_gn_wd_0.1_FINALtest_results_90.json')
    print(np.unique(results['one_hot_cls'], return_counts=True))
    

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
