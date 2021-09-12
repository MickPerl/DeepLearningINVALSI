from os import makedirs, path

import matplotlib.pyplot as plt

import config as cfg

IMAGE_FOLDER = path.join("src", "img", cfg.JOB_NAME, cfg.PROBLEM_TYPE)

makedirs(IMAGE_FOLDER, exist_ok=True)


def plot_accuracy(history: dict):
    f = plt.figure()
    plt.plot(history['acc' if cfg.PROBLEM_TYPE == "classification" else 'bin_acc'])
    plt.plot(history['val_acc' if cfg.PROBLEM_TYPE == "classification" else 'val_bin_acc'])
    plt.title("Accuracy during training")
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'accuracy.png'))

    plt.clf()


def plot_loss(history: dict):
    f = plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title("Loss during training")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'loss.png'))

    plt.clf()


def plot_tp(history: dict):
    f = plt.figure()
    plt.plot(history['tp'])
    plt.plot(history['val_tp'])
    plt.title("True positives during training")
    plt.ylabel('True positives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'tp.png'))

    plt.clf()


def plot_fp(history: dict):
    f = plt.figure()
    plt.plot(history['fp'])
    plt.plot(history['val_fp'])
    plt.title("False positives during training")
    plt.ylabel('False positives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'fp.png'))

    plt.clf()


def plot_tn(history: dict):
    f = plt.figure()
    plt.plot(history['tn'])
    plt.plot(history['val_tn'])
    plt.title("True negatives during training")
    plt.ylabel('True negatives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'tn.png'))

    plt.clf()


def plot_fn(history: dict):
    f = plt.figure()
    plt.plot(history['fn'])
    plt.plot(history['val_fn'])
    plt.title("False negatives during training")
    plt.ylabel('False negatives')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'fn.png'))

    plt.clf()


def plot_precision(history: dict):
    f = plt.figure()
    plt.plot(history['prec'])
    plt.plot(history['val_prec'])
    plt.title("Precision during training")
    plt.ylabel('Precision')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'precision.png'))

    plt.clf()


def plot_recall(history: dict):
    f = plt.figure()
    plt.plot(history['rec'])
    plt.plot(history['val_rec'])
    plt.title("Recall during training")
    plt.ylabel('Recall')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(path.join(IMAGE_FOLDER, 'recall.png'))

    plt.clf()