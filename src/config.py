from tensorflow.keras import optimizers

AP_DATASET_PATH = "../dataset_with_AP.csv"
OUTPUT_ACTIVATION_FUNCTION = "sigmoid"
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # default value 0.001
OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE)
DROPOUT_LAYER = False
DROPOUT_LAYER_RATE = 0.5
EPOCH = 30
SMALL_DATASET = False
