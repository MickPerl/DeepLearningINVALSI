from tensorflow.keras import optimizers

AP_DATASET_PATH = "/home/students/michele.perlino/Downloads/dataset_with_AP.csv" # --dataset nomefile
OUTPUT_ACTIVATION_FUNCTION = "sigmoid" # --activation nomefunzione
BATCH_SIZE = 32 # --batchsize intnum
LEARNING_RATE = 0.001  # --learningrate floatnum
OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE) # --optimizer nomeottimizzatore
DROPOUT_LAYER = False # --dropout
DROPOUT_LAYER_RATE = 0.5 # --dropoutrate floatnum
EPOCH = 50 # --epoch intnum
SMALL_DATASET = False # --small
MODEL = 1
NEURONS = 32