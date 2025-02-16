import random
import numpy as np
import torch

if __name__ == "__main__":
    model = torch.load("saved_models/qnli_sdim30_nsdim30_alpha0.1_VBC.torch")
    print(model)
