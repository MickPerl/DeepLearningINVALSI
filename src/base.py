import sys

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras import metrics
from tensorflow.keras import losses

import config as cfg

"""
Lettura del dataset
"""
dataset = pd.read_csv(cfg.AP_DATASET_PATH, sep=',')

dataset['DROPOUT'] = dataset['DROPOUT'].astype('int64') # Memorizzati come boolean e qui convertiti
dataset['Pon'] = dataset['Pon'].astype('int64') # Memorizzati come boolean e qui convertiti
dataset['sesso'] = dataset['sesso'].astype('int64') # Memorizzati come boolean e qui convertiti

"""
Split
"""
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.2)

"""
Undersampling
"""
class_nodrop = train_dataset[train_dataset['DROPOUT'] == False]  # Sovrarappresentata (== False funziona per casting implicito)
class_drop = train_dataset[train_dataset['DROPOUT'] == True]  # Sottorappresentata (== True funziona per casting implicito)

# Sotto campionamento di class_drop in modo che abbia stessa cardinalità di class_nodrop
class_nodrop = class_nodrop.sample(len(class_drop))

# Ricreazione dataset di test
train_dataset = class_drop.append(class_nodrop)
train_dataset = train_dataset.sample(frac=1)

"""
Convert from pandas DF to tensorflow DS
"""


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.drop("Unnamed: 0", axis=1)
    labels = dataframe.pop("DROPOUT")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels.values))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

if cfg.SMALL_DATASET:
    if input("WARNING: test dataset of size 100. Do you want to proceed? [y/n]") == "y":
        train_dataset = train_dataset.sample(100) # reduce train_ds size
    else:
        print("Closed")
        sys.exit(0)
train_ds = dataframe_to_dataset(train_dataset)
val_ds = dataframe_to_dataset(validation_dataset)
test_ds = dataframe_to_dataset(test_dataset)

"""
Batching
"""
train_ds = train_ds.batch(cfg.BATCH_SIZE, drop_remainder=True) # drop_remainder=True rimuove i record che non rientrano nei batch da 32
val_ds = val_ds.batch(cfg.BATCH_SIZE, drop_remainder=True)
test_ds = test_ds.batch(cfg.BATCH_SIZE, drop_remainder=True)

"""
Encoding feature
"""


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

"""
Input layers
"""
# Categorical features encoded as integers
prog = keras.Input(shape=(1,), name="prog", dtype="int64")
mese = keras.Input(shape=(1,), name="mese", dtype="int64")
anno = keras.Input(shape=(1,), name="anno", dtype="int64")
luogo = keras.Input(shape=(1,), name="luogo", dtype="int64")
eta = keras.Input(shape=(1,), name="eta", dtype="int64")
freq_asilo_nido = keras.Input(shape=(1,), name="freq_asilo_nido", dtype="int64")
freq_scuola_materna = keras.Input(shape=(1,), name="freq_scuola_materna", dtype="int64")
luogo_padre = keras.Input(shape=(1,), name="luogo_padre", dtype="int64")
titolo_padre = keras.Input(shape=(1,), name="titolo_padre", dtype="int64")
prof_padre = keras.Input(shape=(1,), name="prof_padre", dtype="int64")
luogo_madre = keras.Input(shape=(1,), name="luogo_madre", dtype="int64")
titolo_madre = keras.Input(shape=(1,), name="titolo_madre", dtype="int64")
prof_madre = keras.Input(shape=(1,), name="prof_madre", dtype="int64")
regolarita = keras.Input(shape=(1,), name="regolarità", dtype="int64")
cittadinanza = keras.Input(shape=(1,), name="cittadinanza", dtype="int64")
cod_provincia_ISTAT = keras.Input(shape=(1,), name="cod_provincia_ISTAT", dtype="int64")
sigla_provincia_istat = keras.Input(shape=(1,), name="sigla_provincia_istat", dtype="int64")
Nome_reg = keras.Input(shape=(1,), name="Nome_reg", dtype="int64")
Cod_reg = keras.Input(shape=(1,), name="Cod_reg", dtype="int64")
Areageo_3 = keras.Input(shape=(1,), name="Areageo_3", dtype="int64")
Areageo_4 = keras.Input(shape=(1,), name="Areageo_4", dtype="int64")
Areageo_5 = keras.Input(shape=(1,), name="Areageo_5", dtype="int64")
Areageo_5_Istat = keras.Input(shape=(1,), name="Areageo_5_Istat", dtype="int64")
LIVELLI = keras.Input(shape=(1,), name="LIVELLI", dtype="int64")

sesso = keras.Input(shape=(1,), name="sesso", dtype="int64")
Pon = keras.Input(shape=(1,), name="Pon", dtype="int64")

# Numerical features
pu_ma_gr = keras.Input(shape=(1,), name="pu_ma_gr", dtype="float32")
pu_ma_no = keras.Input(shape=(1,), name="pu_ma_no", dtype="float32")
Fattore_correzione_new = keras.Input(shape=(1,), name="Fattore_correzione_new", dtype="float32")
Cheating = keras.Input(shape=(1,), name="Cheating", dtype="float32")
WLE_MAT = keras.Input(shape=(1,), name="WLE_MAT", dtype="float32")
WLE_MAT_200 = keras.Input(shape=(1,), name="WLE_MAT_200", dtype="float32")
WLE_MAT_200_CORR = keras.Input(shape=(1,), name="WLE_MAT_200_CORR", dtype="float32")
pu_ma_no_corr = keras.Input(shape=(1,), name="pu_ma_no_corr", dtype="float32")
n_stud_prev = keras.Input(shape=(1,), name="n_stud_prev", dtype="float32")
Numeri = keras.Input(shape=(1,), name="Numeri", dtype="float32")
n_classi_prev = keras.Input(shape=(1,), name="n_classi_prev", dtype="float32")
Dati = keras.Input(shape=(1,), name="Dati e previsioni", dtype="float32")
Riconoscere_forme = keras.Input(shape=(1,), name="Riconoscere le forme nello spazio e utilizzarle per la risoluzione di problemi geometrici o di modellizzazione", dtype="float32")
Conoscere_padr = keras.Input(shape=(1,), name="Conoscere e padroneggiare i contenuti specifici della matematica", dtype="float32")
Relazioni = keras.Input(shape=(1,), name="Relazioni e funzioni", dtype="float32")
Spazio = keras.Input(shape=(1,), name="Spazio figure", dtype="float32")
Acquisire = keras.Input(shape=(1,), name="Acquisire progressivamente forme tipiche del pensiero matematico", dtype="float32")
Conoscere_util = keras.Input(shape=(1,), name="Conoscere e utilizzare algoritmi e procedure", dtype="float32")
Rappresentare = keras.Input(shape=(1,), name="Rappresentare relazioni e dati e, in situazioni significative, utilizzare le rappresentazioni per ricavare informazioni, formulare giudizi e prendere decisioni", dtype="float32")
Riconoscere_contesti = keras.Input(shape=(1,), name="Riconoscere in contesti diversi il carattere misurabile di oggetti e fenomeni, utilizzare strumenti di misura, misurare grandezze, stimare misure di grandezze", dtype="float32")
Risolvere = keras.Input(shape=(1,), name="Risolvere problemi utilizzando strategie in ambiti diversi – numerico, geometrico, algebrico –", dtype="float32")
Utilizzare = keras.Input(shape=(1,), name="Utilizzare strumenti, modelli e rappresentazioni nel trattamento quantitativo dell'informazione in ambito scientifico, tecnologico, economico e sociale", dtype="float32")

all_inputs = [
    prog,
    sesso,
    mese,
    anno,
    luogo,
    eta,
    freq_asilo_nido,
    freq_scuola_materna,
    luogo_padre,
    titolo_padre,
    prof_padre,
    luogo_madre,
    titolo_madre,
    prof_madre,
    regolarita,
    cittadinanza,
    cod_provincia_ISTAT,
    sigla_provincia_istat,
    Nome_reg,
    Cod_reg,
    Areageo_3,
    Areageo_4,
    Areageo_5,
    Areageo_5_Istat,
    Pon,
    pu_ma_gr,
    pu_ma_no,
    Fattore_correzione_new,
    Cheating,
    WLE_MAT,
    WLE_MAT_200,
    WLE_MAT_200_CORR,
    pu_ma_no_corr,
    n_stud_prev,
    n_classi_prev,
    LIVELLI,
    Numeri,
    Dati,
    Riconoscere_forme,
    Conoscere_padr,
    Relazioni,
    Spazio,
    Acquisire,
    Conoscere_util,
    Rappresentare,
    Riconoscere_contesti,
    Risolvere,
    Utilizzare
]


"""
Encode layers
"""

sesso_encoded = encode_categorical_feature(sesso, "sesso", train_ds, False)
Pon_encoded = encode_categorical_feature(Pon, "Pon", train_ds, False)

n_stud_prev_encoded = encode_numerical_feature(n_stud_prev, "n_stud_prev", train_ds)
n_classi_prev_encoded = encode_numerical_feature(n_classi_prev, "n_classi_prev", train_ds)
pu_ma_gr_encoded = encode_numerical_feature(pu_ma_gr, "pu_ma_gr", train_ds)
pu_ma_no_encoded = encode_numerical_feature(pu_ma_no, "pu_ma_no", train_ds)
Fattore_correzione_new_encoded = encode_numerical_feature(Fattore_correzione_new, "Fattore_correzione_new", train_ds)
Cheating_encoded = encode_numerical_feature(Cheating, "Cheating", train_ds)
WLE_MAT_encoded = encode_numerical_feature(WLE_MAT, "WLE_MAT", train_ds)
WLE_MAT_200_encoded = encode_numerical_feature(WLE_MAT_200, "WLE_MAT_200", train_ds)
WLE_MAT_200_CORR_encoded = encode_numerical_feature(WLE_MAT_200_CORR, "WLE_MAT_200_CORR", train_ds)
pu_ma_no_corr_encoded = encode_numerical_feature(pu_ma_no_corr, "pu_ma_no_corr", train_ds)
Numeri_encoded = encode_numerical_feature(Numeri, "Numeri", train_ds)
Dati_encoded = encode_numerical_feature(Dati, "Dati e previsioni", train_ds)
Riconoscere_forme_encoded = encode_numerical_feature(Riconoscere_forme, "Riconoscere le forme nello spazio e utilizzarle per la risoluzione di problemi geometrici o di modellizzazione", train_ds)
Conoscere_padr_encoded = encode_numerical_feature(Conoscere_padr, "Conoscere e padroneggiare i contenuti specifici della matematica", train_ds)
Relazioni_encoded = encode_numerical_feature(Relazioni, "Relazioni e funzioni", train_ds)
Spazio_encoded = encode_numerical_feature(Spazio, "Spazio figure", train_ds)
Acquisire_encoded = encode_numerical_feature(Acquisire, "Acquisire progressivamente forme tipiche del pensiero matematico", train_ds)
Conoscere_util_encoded = encode_numerical_feature(Conoscere_util, "Conoscere e utilizzare algoritmi e procedure", train_ds)
Rappresentare_encoded = encode_numerical_feature(Rappresentare, "Rappresentare relazioni e dati e, in situazioni significative, utilizzare le rappresentazioni per ricavare informazioni, formulare giudizi e prendere decisioni", train_ds)
Riconoscere_contesti_encoded = encode_numerical_feature(Riconoscere_contesti, "Riconoscere in contesti diversi il carattere misurabile di oggetti e fenomeni, utilizzare strumenti di misura, misurare grandezze, stimare misure di grandezze", train_ds)
Risolvere_encoded = encode_numerical_feature(Risolvere, "Risolvere problemi utilizzando strategie in ambiti diversi – numerico, geometrico, algebrico –", train_ds)
Utilizzare_encoded = encode_numerical_feature(Utilizzare, "Utilizzare strumenti, modelli e rappresentazioni nel trattamento quantitativo dell'informazione in ambito scientifico, tecnologico, economico e sociale", train_ds)

prog_encoded = encode_categorical_feature(prog, "prog", train_ds, False)
mese_encoded = encode_categorical_feature(mese, "mese", train_ds, False)
anno_encoded = encode_categorical_feature(anno, "anno", train_ds, False)
luogo_encoded = encode_categorical_feature(luogo, "luogo", train_ds, False)
eta_encoded = encode_categorical_feature(eta, "eta", train_ds, False)
freq_asilo_nido_encoded = encode_categorical_feature(freq_asilo_nido, "freq_asilo_nido", train_ds, False)
freq_scuola_materna_encoded = encode_categorical_feature(freq_scuola_materna, "freq_scuola_materna", train_ds, False)
luogo_padre_encoded = encode_categorical_feature(luogo_padre, "luogo_padre", train_ds, False)
titolo_padre_encoded = encode_categorical_feature(titolo_padre, "titolo_padre", train_ds, False)
prof_padre_encoded = encode_categorical_feature(prof_padre, "prof_padre", train_ds, False)
luogo_madre_encoded = encode_categorical_feature(luogo_madre, "luogo_madre", train_ds, False)
titolo_madre_encoded = encode_categorical_feature(titolo_madre, "titolo_madre", train_ds, False)
prof_madre_encoded = encode_categorical_feature(prof_madre, "prof_madre", train_ds, False)
regolarita_encoded = encode_categorical_feature(regolarita, "regolarità", train_ds, False)
cittadinanza_encoded = encode_categorical_feature(cittadinanza, "cittadinanza", train_ds, False)
cod_provincia_ISTAT_encoded = encode_categorical_feature(cod_provincia_ISTAT, "cod_provincia_ISTAT", train_ds, False)
sigla_provincia_istat_encoded = encode_categorical_feature(sigla_provincia_istat, "sigla_provincia_istat", train_ds, False)
Nome_reg_encoded = encode_categorical_feature(Nome_reg, "Nome_reg", train_ds, False)
Cod_reg_encoded = encode_categorical_feature(Cod_reg, "Cod_reg", train_ds, False)
Areageo_3_encoded = encode_categorical_feature(Areageo_3, "Areageo_3", train_ds, False)
Areageo_4_encoded = encode_categorical_feature(Areageo_4, "Areageo_4", train_ds, False)
Areageo_5_encoded = encode_categorical_feature(Areageo_5, "Areageo_5", train_ds, False)
Areageo_5_Istat_encoded = encode_categorical_feature(Areageo_5_Istat, "Areageo_5_Istat", train_ds, False)
LIVELLI_encoded = encode_categorical_feature(LIVELLI, "LIVELLI", train_ds, False)


"""
Neural network architecture
"""

all_features = layers.concatenate(
    [
        prog_encoded,
        sesso_encoded,
        mese_encoded,
        anno_encoded,
        luogo_encoded,
        eta_encoded,
        freq_asilo_nido_encoded,
        freq_scuola_materna_encoded,
        luogo_padre_encoded,
        titolo_padre_encoded,
        prof_padre_encoded,
        luogo_madre_encoded,
        titolo_madre_encoded,
        prof_madre_encoded,
        regolarita_encoded,
        cittadinanza_encoded,
        cod_provincia_ISTAT_encoded,
        sigla_provincia_istat_encoded,
        Nome_reg_encoded,
        Cod_reg_encoded,
        Areageo_3_encoded,
        Areageo_4_encoded,
        Areageo_5_encoded,
        Areageo_5_Istat_encoded,
        Pon_encoded,
        pu_ma_gr_encoded,
        pu_ma_no_encoded,
        Fattore_correzione_new_encoded,
        Cheating_encoded,
        WLE_MAT_encoded,
        WLE_MAT_200_encoded,
        WLE_MAT_200_CORR_encoded,
        pu_ma_no_corr_encoded,
        n_stud_prev_encoded,
        n_classi_prev_encoded,
        LIVELLI_encoded,
        Numeri_encoded,
        Dati_encoded,
        Riconoscere_forme_encoded,
        Conoscere_padr_encoded,
        Relazioni_encoded,
        Spazio_encoded,
        Acquisire_encoded,
        Conoscere_util_encoded,
        Rappresentare_encoded,
        Riconoscere_contesti_encoded,
        Risolvere_encoded,
        Utilizzare_encoded
    ]
)

"""
TODO:
- cambiare sigmoid in softmax (forse softmax è sbagliato per la classificazione binaria) -> cfg.OUTPUT_ACTIVATION_FUNCTION
- aggiungere più neuroni (cosa vuol dire? aumentare i batch)
- modificare la dimensione dei batch -> cfg.BATCH_SIZE
- aggiungere più layer
- cambiare ottimizzatore (da Adam a SGD) -> cfg.OPTIMIZER
- cambiare il learning rate dell'ottimizzatore -> cfg.LEARNING_RATE
- rimuovere il layer di dropout -> cfg.DROPOUT_LAYER
- cambiare la possibilità di dropout -> cfg.DROPOUT_LAYER_RATE
"""

x = layers.Dense(32, activation="relu")(all_features)
if cfg.DROPOUT_LAYER:
    x = layers.Dropout(cfg.DROPOUT_LAYER_RATE)(x)
output = layers.Dense(1, activation=cfg.OUTPUT_ACTIVATION_FUNCTION)(x)
model = keras.Model(all_inputs, output)

model.compile(optimizer=cfg.OPTIMIZER,
              loss=losses.BinaryCrossentropy(),
              metrics=[metrics.Accuracy(),
                       metrics.Precision(),
                       metrics.Recall(),
                       metrics.FalseNegatives(),
                       metrics.FalsePositives(),
                       metrics.TrueNegatives(),
                       metrics.TruePositives()])

"""
Training
"""

model.fit(train_ds, epochs=cfg.EPOCH, validation_data=val_ds)

"""
Evaluation
"""
score = model.evaluate(test_ds)
print('Test loss:', score[0])
print('Test accuracy:', score[1])