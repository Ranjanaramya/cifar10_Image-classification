# utils.py - helpers for plotting and metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def plot_cm(y_true, y_pred, out_path='plots/cm.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(out_path)
    plt.close()

def print_classification(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
