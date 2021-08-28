from tensorflow.keras import optimizers

# base.py
AP_DATASET_PATH = "/Users/marco/Documents/Università/Intelligenza artificiale/project.nosync/MachineLearningProject/dataset_with_AP.csv" # --dataset nomefile
SMALL_DATASET = False # --small
MODEL = 2
# both
LEARNING_RATE = 0.001  # --learningrate floatnum
OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE) # --optimizer nomeottimizzatore
DROPOUT_LAYER = False # --dropout
DROPOUT_LAYER_RATE = 0.5 # --dropoutrate floatnum
EPOCH = 50 # --epoch intnum
NEURONS = 128
OUTPUT_ACTIVATION_FUNCTION = "sigmoid" # --activation nomefunzione
BATCH_SIZE = 32 # --batchsize intnum
# invalsi.py
ORIGINAL_DATASET = "../nuovi_dataset/original_dataset.csv"
CLEANED_DATASET = "../nuovi_dataset/cleaned_dataset.csv"
CLEANED_DATASET_WITH_AP = "../nuovi_dataset/dataset_ap.csv"
UNDERSAMPLED_DATASET = "../nuovi_dataset/undersampled_dataset.csv"
TEST_SET_PERCENT = 0.2
VALIDATION_SET_PERCENT = 0.2
NUMBER_OF_LAYERS = 10