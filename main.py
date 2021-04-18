import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from tqdm import tqdm
import yaml

from datasets import CaltransTraffic, split_dataset

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    config_path = "configs/config_example.yaml"
    with open(config_path) as file:
        config = yaml.safe_load(file)
    print(f"Loading config {config_path.split('/')[-1]} ...")

    print(f"Loading dataset {config['dataset']} ...")



if __name__ == '__main__':
    main()