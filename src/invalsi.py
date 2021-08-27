import re
from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import config as cfg
from mapping_domande_ambiti_processi import MAPPING_DOMANDE_AMBITI_PROCESSI
from column_converters import COLUMN_CONVERTERS

SKIP = True

"""
Import del dataset originale
"""
if SKIP:
    original_dataset = pd.read_csv(cfg.ORIGINAL_DATASET, sep=';', converters=COLUMN_CONVERTERS)

"""
Cerchiamo colonne che abbiamo percentuali di valori nulli.
"""
if SKIP:
    print("Columns with high null values percentages:")
    for col in original_dataset.columns:
        null_values_mean = original_dataset[col].isnull().mean()
        if null_values_mean > 0:
            print(col, '\t\tType: ', original_dataset[col].dtypes, '\tMissing values:',
                  original_dataset[col].isnull().mean().round(3))

columns_with_high_null_values = ["codice_orario", "PesoClasse", "PesoScuola", "PesoTotale_Matematica"]
columns_with_lower_null_values = [
    "voto_scritto_ita",  # 0.683
    "voto_scritto_mat",  # 0.113
    "voto_orale_ita",  # 0.683
    "voto_orale_mat"  # 0.114
]

"""
Cerchiamo colonne con valori univoci o quasi (ad esempio identificativi).
Se ce ne sono, meglio toglierle perché sono inutili.
"""
if SKIP:
    dataset_len = len(original_dataset)
    print("Columns with unique values:")
    for col in original_dataset.columns:
        unique_vals = original_dataset[col].nunique()
        if unique_vals / dataset_len > 0.1:
            print(col, "ratio = ", round(unique_vals / dataset_len, 3))
columns_with_unique_values = ["Unnamed: 0", "CODICE_STUDENTE"]

"""
Cerchiamo colonne con sempre lo stesso valore perché non danno informazioni.
Se ce ne sono, meglio toglierle perché sono inutili.
"""
if SKIP:
    print("Columns with just one value:")
    for col in original_dataset.columns:
        unique_vals = original_dataset[col].nunique()
        if unique_vals == 1:
            print(col)

columns_with_just_one_value = ["macrotipologia", "livello"]

"""
Rimozione delle colonne indicate in:
- columns_with_high_null_values
- columns_with_lower_null_values (per ora vengono tenute poiché forse possono essere utili)
- columns_with_unique_values
- columns_with_just_one_value
"""
if SKIP:
    cleaned_original_dataset: pd.DataFrame = original_dataset.drop(
        columns_with_high_null_values + columns_with_unique_values + columns_with_just_one_value, axis=1)

    save_cleaned_dataset = False
    if save_cleaned_dataset:
        cleaned_original_dataset.to_csv(cfg.CLEANED_DATASET)
    else:
        cleaned_original_dataset = pd.read_csv(cfg.CLEANED_DATASET)

"""
Creazione lista con domande
"""
if SKIP:
    questions_columns = [col for col in list(cleaned_original_dataset) if re.search("^D\d", col)]

"""
Mapping domande -> (ambiti, processi)
Tutte le colonne delle domande vengono sostituite da colonne ambiti e processi.
"""

list_ambiti_processi = [AP for val in MAPPING_DOMANDE_AMBITI_PROCESSI.values() for AP in val]
ambiti_processi = set(list_ambiti_processi)
conteggio_ambiti_processi = {AP: list_ambiti_processi.count(AP) for AP in ambiti_processi}

if SKIP:
    dataset_with_ambiti_processi = cleaned_original_dataset.copy()
    for AP in ambiti_processi:
        # Aggiunge una colonna al dataset chiamata AP e inizializza tutti i record a 0.0
        dataset_with_ambiti_processi[AP] = 0.0

"""
Per ogni domanda vado a vedere se lo studente ha risposto correttamente o erroneamente:
- se ha risposto correttamente, per ogni ambito o processo vado ad incrementare il valore contenuto nella cella relativa
all'ambito o al processo. L'incremento è di 1/(#domande con quell'ambito o processo).
- se ha risposto erroneamente, non incremento il valore.
Di conseguenza uno studente che ha risposto sempre correttamente a domande di un certo ambito/processo avrà il valore di quella cella a 1.
"""
if SKIP:
    convert_domande_to_ambiti_processi = False

    if convert_domande_to_ambiti_processi:
        for i, row in dataset_with_ambiti_processi.iterrows():
            for question, APs in MAPPING_DOMANDE_AMBITI_PROCESSI.items():
                if row[question] == True:
                    for AP in APs:
                        dataset_with_ambiti_processi.at[i, AP] += 1 / conteggio_ambiti_processi[AP]

        dataset_ap = dataset_with_ambiti_processi.drop(questions_columns, axis=1)

        dataset_ap.to_csv(cfg.CLEANED_DATASET_WITH_AP)
    else:
        dataset_ap = pd.read_csv(cfg.CLEANED_DATASET_WITH_AP)

"""
Scopriamo se ci sono colonne con valori molto correlati.
"""
if SKIP:
    corr_matrix = dataset_ap.corr(method='pearson').round(2)
    upper_corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    upper_corr_matrix.style.background_gradient(cmap='YlOrRd')

interisting_to_check_if_correlated_columns = [
                                                 # Alta correlazione fra voti della stessa materia, abbastanza correlate fra materie diverse
                                                 "voto_scritto_ita",
                                                 "voto_orale_ita",
                                                 "voto_scritto_mat",
                                                 "voto_orale_mat",
                                                 # Correlazione totale, abbastanza correlate con voti
                                                 "pu_ma_gr",
                                                 "pu_ma_no"
                                             ] + list(ambiti_processi)

if SKIP:
    check_corr_dataset = dataset_ap[interisting_to_check_if_correlated_columns].corr(method='pearson').round(2)

    check_corr_dataset.style.background_gradient(cmap='YlOrRd')

"""
Rimozione colonne con alta correlazione.
"""
# Sarà fatto?

"""
Verifica sbilanciamento classi DROPOUT e NO DROPOUT nel dataset.
"""
if SKIP:
    nr_nodrop, nr_drop = np.bincount(dataset_ap['DROPOUT'])
    total_records = nr_drop + nr_nodrop
    print(
        f"Total number of records: {total_records} - \
    Total num. DROPOUT: {nr_drop} - \
    Total num. NO DROPOUT: {nr_nodrop} - \
    Ratio DROPOUT/TOTAL: {round(nr_drop / total_records, 2)} - \
    Ratio NO DROPOUT/TOTAL: {round(nr_nodrop / total_records, 2)} - \
    Ratio DROPOUT/NO DROPOUT: {round(nr_drop / nr_nodrop, 2)}"
    )

"""
Random undersampling
"""
perform_random_undersampling = False
load_random_undersampled_dataset = True
if perform_random_undersampling:
    # class_nodrop contiene i record della classe sovrarappresentata, ovvero SENZA DROPOUT.
    class_nodrop = dataset_ap[dataset_ap['DROPOUT'] == False]
    # class_drop contiene i record della classe sottorappresentata, ovvero CON DROPOUT.
    class_drop = dataset_ap[dataset_ap['DROPOUT'] == True]

    # Sotto campionamento di class_drop in modo che abbia stessa cardinalità di class_nodrop
    class_nodrop = class_nodrop.sample(len(class_drop))

    print(f'Class NO DROPOUT: {len(class_nodrop):,}')
    print(f'Classe DROPOUT: {len(class_drop):,}')

    sampled_dataset = class_drop.append(class_nodrop)
    sampled_dataset = sampled_dataset.sample(frac=1)

    sampled_dataset.to_csv(cfg.UNDERSAMPLED_DATASET)
elif load_random_undersampled_dataset:
    sampled_dataset = pd.read_csv(cfg.UNDERSAMPLED_DATASET)
else:
    sampled_dataset = dataset_ap.copy()

"""
TODO: move out from here
"""
sampled_dataset["sigla_provincia_istat"].fillna(value="ND", inplace=True)

sampled_dataset["voto_scritto_ita"].fillna(value=sampled_dataset["voto_scritto_ita"].mean(), inplace=True)
sampled_dataset["voto_orale_ita"].fillna(value=sampled_dataset["voto_orale_ita"].mean(), inplace=True)
sampled_dataset["voto_scritto_mat"].fillna(value=sampled_dataset["voto_scritto_mat"].mean(), inplace=True)
sampled_dataset["voto_orale_mat"].fillna(value=sampled_dataset["voto_orale_mat"].mean(), inplace=True)
print(sampled_dataset.isna().any())

"""
Comprensione tipi colonne per trovare:
- lista feature continue (float)
- lista feature ordinali (int)
- lista feature categoriche intere (int)
- lista feature categoriche stringhe (str)
- lista feature binarie
"""
print("Lista colonne e tipi:")
print(sampled_dataset.info())

continuous_features = columns_with_lower_null_values + \
                      ["pu_ma_no", "Fattore_correzione_new", "Cheating", "WLE_MAT", "WLE_MAT_200", "WLE_MAT_200_CORR",
                       "pu_ma_no_corr"] + \
                      list(ambiti_processi)
ordinal_features = [
    "n_stud_prev", "n_classi_prev", "LIVELLI", "pu_ma_gr"
]
int_categorical_features = [
    "CODICE_SCUOLA", "CODICE_PLESSO", "CODICE_CLASSE", "campione", "prog",
]
str_categorical_features = [
    "sesso", "mese", "anno", "luogo", "eta", "freq_asilo_nido", "freq_scuola_materna",
    "luogo_padre", "titolo_padre", "prof_padre", "luogo_madre", "titolo_madre", "prof_madre",
    "regolarità", "cittadinanza", "cod_provincia_ISTAT", "Nome_reg",
    "Cod_reg", "Areageo_3", "Areageo_4", "Areageo_5", "Areageo_5_Istat"
]
bool_features = ["Pon"]

"""
Oversampling con SMOTE
"""
# verrà fatto?


"""
Suddivisione dataset in training, validation, test.
"""
df_training_set, df_test_set = train_test_split(sampled_dataset, test_size=cfg.TEST_SET_PERCENT)
df_training_set, df_validation_set = train_test_split(df_training_set, test_size=cfg.VALIDATION_SET_PERCENT)

"""
Conversione da Pandas DataFrame a Tensorflow Dataset.
"""


def dataframe_to_dataset(dataframe: pd.DataFrame):
    copied_df = dataframe.copy()
    # copied_df["sigla_provincia_istat"] = copied_df["sigla_provincia_istat"].fillna("Non disponibile")
    copied_df["DROPOUT"] = copied_df["DROPOUT"].astype("int64")
    dropout_col = copied_df.pop("DROPOUT")
    """
    Dato che il dataframe ha dati eterogenei lo convertiamo a dizionario,
    in cui le chiavi sono i nomi delle colonne e i valori sono i valori della colonna.
    Infine bisogna indicare la colonna target.
    """
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(copied_df), dropout_col))
    tf_dataset = tf_dataset.shuffle(buffer_size=len(copied_df))
    return tf_dataset


ds_training_set = dataframe_to_dataset(df_training_set)
ds_validation_set = dataframe_to_dataset(df_validation_set)
ds_test_set = dataframe_to_dataset(df_test_set)

"""
Suddivisione dei Dataset in batch per sfruttare meglio le capacità hardware
(invece di elaborare un record per volta).
"""
# drop_remainder=True rimuove i record che non rientrano nei batch della dimensione fissata.
ds_training_set = ds_training_set.batch(cfg.BATCH_SIZE, drop_remainder=True)  #
ds_validation_set = ds_validation_set.batch(cfg.BATCH_SIZE, drop_remainder=True)
ds_test_set = ds_test_set.batch(cfg.BATCH_SIZE, drop_remainder=True)

"""
Creazione layer di input per ogni feature a partire dalle liste precedentemente definite:
- continuous_features
- ordinal_features
- int_categorical_features
- str_categorical_features
- bool_features
"""
input_layers = {}
for name, column in df_training_set.items():
    if name != "DROPOUT":
        if name in continuous_features:
            dtype = tf.float32
        elif name in ordinal_features or name in int_categorical_features or name in bool_features:
            dtype = tf.int64
        else:  # str_categorical_features
            dtype = tf.string

        input_layers[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

"""
Encoding delle feature in base al loro tipo.
"""
preprocessed_features = []


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


# Preprocessing colonne con dati booleani
for name in bool_features:
    inp = input_layers[name]
    inp = inp[:, tf.newaxis]
    float_value = tf.cast(inp, tf.float32)
    preprocessed_features.append(float_value)

# Preprocessing colonne con dati interi ordinali
ordinal_inputs = {}
for name in ordinal_features:
    ordinal_inputs[name] = input_layers[name]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(df_training_set[ordinal_features])))
ordinal_inputs = stack_dict(ordinal_inputs)
ordinal_normalized = normalizer(ordinal_inputs)
preprocessed_features.append(ordinal_normalized)

# Preprocessing colonne con dati continui
continuous_inputs = {}
for name in continuous_features:
    continuous_inputs[name] = input_layers[name]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(df_training_set[continuous_features])))
continuous_inputs = stack_dict(continuous_inputs)
continuous_normalized = normalizer(continuous_inputs)
preprocessed_features.append(continuous_normalized)

# Preprocessing colonne con dati categorici stringa
for name in str_categorical_features:
    vocab = sorted(set(df_training_set[name]))

    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')

    x = input_layers[name][:, tf.newaxis]
    x = lookup(x)

    preprocessed_features.append(x)

# Preprocessing colonne con dati categorici interi
for name in int_categorical_features:
    vocab = sorted(set(df_training_set[name]))

    lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

    x = input_layers[name][:, tf.newaxis]
    x = lookup(x)

    preprocessed_features.append(x)

"""
Assemblaggio dei vari layer preprocessati.
"""
# initializer = tf.keras.initializers.glorot_uniform(seed=19)

preprocessed = tf.concat(preprocessed_features, axis=-1)

preprocessor = tf.keras.Model(input_layers, preprocessed)

body = tf.keras.Sequential(
    [tf.keras.layers.Dense(cfg.NEURONS, activation="relu") for _ in range(cfg.NUMBERO_OF_LAYERS)] +
    [tf.keras.layers.Dense(1, activation="sigmoid")]
)

x = preprocessor(input_layers)

result = body(x)

model = tf.keras.Model(input_layers, result)

model.compile(optimizer=cfg.OPTIMIZER,
              loss=tf.losses.BinaryCrossentropy(),
              metrics=[
                  tf.metrics.Accuracy(),
                  tf.metrics.BinaryAccuracy(),
                  tf.metrics.Precision(),
                  tf.metrics.Recall(),
                  tf.metrics.FalseNegatives(),
                  tf.metrics.FalsePositives(),
                  tf.metrics.TrueNegatives(),
                  tf.metrics.TruePositives()])

model.fit(ds_training_set, epochs=cfg.EPOCH, batch_size=cfg.BATCH_SIZE, validation_data=ds_validation_set, verbose=2)
