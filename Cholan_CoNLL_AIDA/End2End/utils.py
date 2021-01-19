import tensorflow as tf
import torch
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, confusion_matrix
import os
import logging
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

## Default file and trained model location ##
#data_dir = "/data/prabhakar/CG/NED_data/"
#data_dir = "/data/prabhakar/CG/NED_data/without_localcontext/"
#test_data_dir = data_dir + "data_test_5/"
#output_dir = "/data/prabhakar/CG/NED_pretrained/model_data_50000/"
#output_dir = "/data/prabhakar/CG/NED_pretrained/without_localcontext/"

## CUDA devices ##
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print('GPU available -', device)
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

## Hyperparameters ##
MAX_LEN = 32
batch_size = 4
epochs = 4

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_loss(loss_values, label=None):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o')

    # Label the plot.
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(output_dir+label + '.png')
    plt.show()


def compute_metrics(preds_flat, labels_flat):
    assert len(preds_flat) == len(labels_flat)
    matrix_results = confusion_matrix(preds_flat, labels_flat)
    accuracy = accuracy_score(preds_flat, labels_flat)
    report = classification_report(preds_flat, labels_flat)

    print('Confusion Matrix :', matrix_results)
    print('Accuracy Score :', accuracy)
    print('Report : ', report)

    return matrix_results, accuracy, report
