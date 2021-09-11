from os import makedirs, path

import matplotlib.pyplot as plt

import config as cfg

IMAGE_FOLDER = path.join("src", "img")

makedirs(path.join(IMAGE_FOLDER, cfg.JOB_NAME, cfg.PROBLEM_TYPE), exist_ok=True)

def plot_accuracy(history):
    plt.plot(history.history['acc' if cfg.PROBLEM_TYPE == "classification" else 'bin_acc'])
    plt.plot(history.history['val_acc' if cfg.PROBLEM_TYPE == "classification" else 'val_bin_acc'])
    plt.title("Accuracy during training")
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, cfg.JOB_NAME, 'accuracy.png'))

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss during training")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, cfg.JOB_NAME, 'loss.png'))

def plot_tp(history):
    plt.plot(history.history['tp'])
    plt.plot(history.history['val_tp'])
    plt.title("True positives during training")
    plt.ylabel('True positives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, cfg.JOB_NAME, 'tp.png'))

def plot_fp(history):
    plt.plot(history.history['fp'])
    plt.plot(history.history['val_fp'])
    plt.title("False positives during training")
    plt.ylabel('False positives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, cfg.JOB_NAME, 'fp.png'))

def plot_tn(history):
    plt.plot(history.history['tn'])
    plt.plot(history.history['val_tn'])
    plt.title("True negatives during training")
    plt.ylabel('True negatives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, cfg.JOB_NAME, 'tn.png'))

def plot_fn(history):
    plt.plot(history.history['fn'])
    plt.plot(history.history['val_fn'])
    plt.title("False positives during training")
    plt.ylabel('False negatives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, cfg.JOB_NAME, 'fn.png'))