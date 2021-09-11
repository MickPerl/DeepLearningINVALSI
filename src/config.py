from logging import error
from tensorflow.keras import optimizers
from os import getenv

# both
LEARNING_RATE = float(getenv(key="LEARNING_RATE", default="0.001"))  # --learningrate floatnum
DROPOUT_LAYER = eval(getenv(key="DROPOUT_LAYER", default="False")) # --dropout
EPOCH = int(getenv(key="EPOCH", default="50")) # --epoch intnum
NEURONS = int(getenv(key="NEURONS", default="128")) # --neurons intnum
BATCH_SIZE = int(getenv(key="BATCH_SIZE", default="32")) # --batchsize intnum
# base.py
OUTPUT_ACTIVATION_FUNCTION = getenv(key="OUTPUT_ACTIVATION_FUNCTION", default="softmax") # --activation nomefunzione
DENSE_LAYER_ACTIVATION = getenv(key="DENSE_LAYER_ACTIVATION", default="relu") # hidden layers' activation function 
AP_DATASET_PATH = getenv(key="AP_DATASET_PATH", default="/Users/marco/Documents/Università/Intelligenza artificiale/project.nosync/MachineLearningProject/dataset_with_AP.csv") # --dataset nomefile
SMALL_DATASET = eval(getenv(key="SMALL_DATASET", default="False")) # --small
MODEL = int(getenv(key="MODEL", default="1")) # --model intnum
OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE) # --optimizer nomeottimizzatore
# invalsi.py
ORIGINAL_DATASET = getenv(key="ORIGINAL_DATASET", default="../nuovi_dataset/original_dataset.csv")
CLEANED_DATASET = getenv(key="CLEANED_DATASET", default="../nuovi_dataset/cleaned_dataset.csv")
CLEANED_DATASET_WITH_AP = getenv(key="CLEANED_DATASET_WITH_AP", default="../nuovi_dataset/dataset_ap.csv")
SAMPLING_TO_PERFORM = getenv(key="SAMPLING_TO_PERFORM", default="random_undersampling")
TEST_SET_PERCENT = float(getenv(key="TEST_SET_PERCENT", default="0.2"))
VALIDATION_SET_PERCENT = float(getenv(key="VALIDATION_SET_PERCENT", default="0.2"))
NUMBER_OF_LAYERS = int(getenv(key="NUMBER_OF_LAYERS", default="10"))
FILL_NAN = getenv(key="FILL_NAN", default="median")
ACTIVATION_LAYER = getenv(key="ACTIVATION_LAYER", default="leaky_relu")
EARLY_STOPPING = eval(getenv(key="EARLY_STOPPING", default="True"))
PROBLEM_TYPE = getenv(key="PROBLEM_TYPE", default="classification")
JOB_NAME = getenv(key="JOB_NAME", default="default")
DROPOUT_HIDDEN_LAYER_RATE = float(getenv(key="DROPOUT_HIDDEN_LAYER_RATE", default="0.5"))
DROPOUT_INPUT_LAYER_RATE = float(getenv(key="DROPOUT_INPUT_LAYER_RATE", default="0.8"))

def print_config():
    print("JOB_NAME", JOB_NAME)
    print("PROBLEM_TYPE: ", PROBLEM_TYPE)
    print("LEARNING_RATE: ", LEARNING_RATE)
    print("DROPOUT_LAYER: ", DROPOUT_LAYER)
    print("DROPOUT_INPUT_LAYER_RATE: ", DROPOUT_INPUT_LAYER_RATE)
    print("DROPOUT_HIDDEN_LAYER_RATE: ", DROPOUT_HIDDEN_LAYER_RATE)
    print("EPOCH: ", EPOCH)
    print("NEURONS: ", NEURONS)
    print("BATCH_SIZE: ", BATCH_SIZE)
    print("ORIGINAL_DATASET: ", ORIGINAL_DATASET)
    print("CLEANED_DATASET: ", CLEANED_DATASET)
    print("CLEANED_DATASET_WITH_AP: ", CLEANED_DATASET_WITH_AP)
    print("SAMPLING_TO_PERFORM: ", SAMPLING_TO_PERFORM)
    print("TEST_SET_PERCENT: ", TEST_SET_PERCENT)
    print("VALIDATION_SET_PERCENT: ", VALIDATION_SET_PERCENT)
    print("NUMBER_OF_LAYERS: ", NUMBER_OF_LAYERS)
    print("FILL_NAN: ", FILL_NAN)
    print("ACTIVATION_LAYER: ", ACTIVATION_LAYER)
    print("EARLY_STOPPING: ", EARLY_STOPPING)


def check_config():
    errors = 0
    if PROBLEM_TYPE not in ["classification", "regression"]:
        print("PROBLEM_TYPE should be either \"classification\" or \"regression\".")
        errors += 1
    
    if LEARNING_RATE < 0 or LEARNING_RATE > 1:
        print("LEARNING_RATE should be in range [0..1].")
        errors += 1

    if DROPOUT_INPUT_LAYER_RATE < 0 or DROPOUT_INPUT_LAYER_RATE > 1:
        print("DROPOUT_INPUT_LAYER_RATE should be in range [0..1].")
        errors += 1

    if DROPOUT_HIDDEN_LAYER_RATE < 0 or DROPOUT_HIDDEN_LAYER_RATE > 1:
        print("DROPOUT_HIDDEN_LAYER_RATE should be in range [0..1].")
        errors += 1

    if EPOCH < 1:
        print("EPOCH should be greater than 0.")
        errors += 1
    
    if NEURONS < 1:
        print("NEURONS should be greater than 0.")
        errors += 1
    
    if BATCH_SIZE < 1:
        print("BATCH_SIZE should be greater than 0.")
        errors += 1
    
    if SAMPLING_TO_PERFORM not in ["random_undersampling", "SMOTE"]:
        print("SAMPLING_TO_PERFORM should either be \"random_undersampling\" or \"SMOTE\".")
        errors += 1

    if TEST_SET_PERCENT < 0 or TEST_SET_PERCENT > 1:
        print("TEST_SET_PERCENT should be in range [0..1].")
        errors += 1
    
    if VALIDATION_SET_PERCENT < 0 or VALIDATION_SET_PERCENT > 1:
        print("VALIDATION_SET_PERCENT should be in range [0..1].")
        errors += 1
    
    if NUMBER_OF_LAYERS < 1:
        print("NUMBER_OF_LAYERS should be greater than 0.")
        errors += 1
    
    if FILL_NAN not in ["median", "mean", "remove"]:
        print("FILL_NAN should be \"median\", \"mean\" or \"remove\".")
        errors += 1
    
    if ACTIVATION_LAYER not in ["relu", "leaky_relu"]:
        print("ACTIVATION_LAYER should either be \"relu\" or \"leaky_relu\".")
        errors += 1