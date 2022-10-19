import pandas as pd
import os
DATA_PATH="./dataset"

SPI = pd.read_csv(os.path.join(DATA_PATH,"SPI_training_2.csv"))
AOI = pd.read_csv(os.path.join(DATA_PATH,"AOI_training.csv"))
print("")