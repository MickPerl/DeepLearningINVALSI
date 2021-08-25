import re
from typing import List

import pandas as pd
import numpy as np

import config as cfg
from mapping_domande_ambiti_processi import MAPPING_DOMANDE_AMBITI_PROCESSI
from column_converters import COLUMN_CONVERTERS

"""
Import del dataset originale
"""
original_dataset = pd.read_csv(cfg.ORIGINAL_DATASET, sep=';', converters=COLUMN_CONVERTERS)


"""
Cerchiamo colonne che abbiamo percentuali di valori nulli.
"""
print("Columns with high null values percentages:")
for col in original_dataset.columns:
    null_values_mean = original_dataset[col].isnull().mean()
    if null_values_mean > 0:
        print(col, '\t\tType: ', original_dataset[col].dtypes, '\tMissing values:', original_dataset[col].isnull().mean().round(3))

columns_with_high_null_values = ["codice_orario", "PesoClasse", "PesoScuola", "PesoTotale_Matematica"]
columns_with_lower_null_values = [
    "voto_scritto_ita", # 0.683
    "voto_scritto_mat",# 0.113
    "voto_orale_ita", # 0.683
    "voto_orale_mat" # 0.114
]

"""
Cerchiamo colonne con valori univoci o quasi (ad esempio identificativi).
Se ce ne sono, meglio toglierle perché sono inutili.
"""
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
cleaned_original_dataset: pd.DataFrame = original_dataset.drop(columns_with_high_null_values + columns_with_unique_values + columns_with_just_one_value, axis=1)

save_cleaned_dataset = True
if save_cleaned_dataset:
    cleaned_original_dataset.to_csv(cfg.CLEANED_DATASET)
else:
    cleaned_original_dataset = pd.read_csv(cfg.CLEANED_DATASET)

"""
Creazione lista con domande
"""
questions_columns = [col for col in list(cleaned_original_dataset) if re.search("^D\d", col)]
# questions_dataset = original_dataset[questions_columns]


"""
Mapping domande -> (ambiti, processi)
Tutte le colonne delle domande vengono sostituite da colonne ambiti e processi.
"""

list_ambiti_processi = [AP for val in MAPPING_DOMANDE_AMBITI_PROCESSI.values() for AP in val]
ambiti_processi = set(list_ambiti_processi)
conteggio_ambiti_processi = {AP: list_ambiti_processi.count(AP) for AP in ambiti_processi}

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

convert_domande_to_ambiti_processi = True

if convert_domande_to_ambiti_processi:
    for i, row in dataset_with_ambiti_processi.iterrows():
        for question, APs in MAPPING_DOMANDE_AMBITI_PROCESSI.items():
            if row[question] == True:
                for AP in APs:
                    dataset_with_ambiti_processi.at[i, AP] += 1/conteggio_ambiti_processi[AP]

    dataset_ap = dataset_with_ambiti_processi.drop(questions_columns, axis=1)

    dataset_ap.to_csv(cfg.CLEANED_DATASET_WITH_AP)
else:
    dataset_ap = pd.read_csv(cfg.CLEANED_DATASET_WITH_AP)

"""
Scopriamo se ci sono colonne con valori molto correlati.
"""
corr_matrix = dataset_ap.corr(method='pearson').round(2)
upper_corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper_corr_matrix.style.background_gradient(cmap='YlOrRd')