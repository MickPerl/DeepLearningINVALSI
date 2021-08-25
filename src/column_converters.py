from typing import List, Dict, Tuple
string_list = List[str]
one_hot_list = Tuple[int]
one_hot_encoding_list = Dict[str, one_hot_list]
one_hot_encoding_int = Dict[str, int]
import numpy as np

def from_categorical_to_one_hot_int(categorical_data: string_list) -> one_hot_encoding_int:
    dictionary_to_return = {}
    for index, key in enumerate(categorical_data):
        dictionary_to_return[key] = index
    
    return dictionary_to_return

def from_categorical_to_one_hot_list(categorical_data: string_list) -> one_hot_encoding_list:
    dictionary_to_return = {}
    indexes = range(len(categorical_data))
    for index, key in enumerate(categorical_data):
        dictionary_to_return[key] = [1 if index == i else 0 for i in indexes]
    
    return dictionary_to_return

#Features categorica -> One Hot Encoding
list_MESI = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre", "Non disponibile"]
MESI = from_categorical_to_one_hot_int(list_MESI)

#Features categorica -> One Hot Encoding
list_ANNI = ["2000", "2001", "1999", "1998", "<=1997", ">=2002", "Non disponibile"]
ANNI = from_categorical_to_one_hot_int(list_ANNI)

#Features categorica -> One Hot Encoding
list_ETA = ['Mancante di sistema', 'Non disponibile', '6 anni', '5 anni', '10 anni o più', '2 anni', '1 anno o prima', '9 anni', '8 anni', '4 anni', '7 anni', '3 anni']
ETA = from_categorical_to_one_hot_int(list_ETA)

def convert_question_result(result: str) -> bool:
    return result == "Corretta" # alternativamente result == "Errata"

#Features categorica -> One Hot Encoding
list_REGOLARITA = ['Regolare', 'Posticipatario', 'Anticipatario', 'Dato mancante']
REGOLARITA = from_categorical_to_one_hot_int(list_REGOLARITA)

#Features categorica -> One Hot Encoding
list_AREA_GEOGRAFICA_5_ISTAT = ['Sud', 'Nord est', 'Centro', 'Nord ovest', 'Isole']
AREA_GEOGRAFICA_5_ISTAT = from_categorical_to_one_hot_int(list_AREA_GEOGRAFICA_5_ISTAT)

#Features categorica -> One Hot Encoding
list_AREA_GEOGRAFICA_5 = ['Sud', 'Nord est', 'Centro', 'Nord ovest', 'Sud e isole']
AREA_GEOGRAFICA_5 = from_categorical_to_one_hot_int(list_AREA_GEOGRAFICA_5)

#Features categorica -> One Hot Encoding
list_AREA_GEOGRAFICA_4 = ['Mezzogiorno', 'Nord est', 'Centro', 'Nord ovest']
AREA_GEOGRAFICA_4 = from_categorical_to_one_hot_int(list_AREA_GEOGRAFICA_4)

#Features categorica -> One Hot Encoding
list_AREA_GEOGRAFICA_3 = ['Mezzogiorno', 'Nord', 'Centro']
AREA_GEOGRAFICA_3 = from_categorical_to_one_hot_int(list_AREA_GEOGRAFICA_3)

#Features categorica -> One Hot Encoding
list_REGIONI = ['Campania', 'Emilia-Romagna', 'Lazio', 'Piemonte', 'Puglia', 'Lombardia', 'Veneto', 'Sicilia', 'Prov. Aut. Trento', 'Friuli-Venezia Giulia', 'Abruzzo', 'Liguria', 'Toscana', 'Sardegna', 'Calabria', 'Molise', 'Marche', 'Umbria', 'Basilicata', 'Prov. Aut. Bolzano (l. it.)']
REGIONI = from_categorical_to_one_hot_int(list_REGIONI)

#Features categorica -> One Hot Encoding
list_PROVINCE = ['', 'RE', 'FR', 'TO', 'BA', 'CO', 'LE', 'RO', 'CT', 'RM', 'TA', 'BS', 'SA', 'TN', 'UD', 'FG', 'LT', 'AG', 'CH', 'PC', 'TS', 'SR', 'SP', 'PD', 'SI', 'PA', 'TP', 'BO', 'CA', 'CN', 'RC', 'TE', 'MI', 'LC', 'LU', 'FI', 'AQ', 'TV', 'RG', 'VA', 'GO', 'MO', 'GE', 'AL', 'CB', 'PR', 'OR', 'VE', 'MC', 'NO', 'PT', 'MN', 'VR', 'PI', 'AP', 'LO', 'VI', 'SV', 'PU', 'BG', 'AR', 'VT', 'LI', 'SS', 'BR', 'RA', 'TR', 'SO', 'IM', 'PZ', 'GR', 'AN', 'PN', 'ME', 'CR', 'FE', 'BI', 'PV', 'PG', 'VB', 'BL', 'PE', 'CS', 'CZ', 'AV', 'RN', 'CL', 'AT', 'MS', 'KR', 'RI', 'EN', 'CE', 'MT', 'VV', 'VC', 'NU', 'FC', 'PO', 'BZ', 'BN', 'IS', 
            'NA', # presente in cod_provincia_ISTAT ma non in sigla_provincia_istat
            'PS', # presente in cod_provincia_ISTAT ma non in sigla_provincia_istat
            'FO', # presente in cod_provincia_ISTAT ma non in sigla_provincia_istat
            'LB', # presente in cod_provincia_ISTAT ma non in sigla_provincia_istat
]
PROVINCE = from_categorical_to_one_hot_int(list_PROVINCE)

#Features categorica -> One Hot Encoding
list_CITTADINANZA = ['Italiano', 'Straniero II generazione', 'Straniero I generazione', 'Dato mancante']
CITTADINANZA = from_categorical_to_one_hot_int(list_CITTADINANZA)

list_VOTI_NUMERICI = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
def voto_orale_decode(voto_orale: str) -> float:
    if voto_orale in list_VOTI_NUMERICI:
        return float(voto_orale)
    elif voto_orale == 'Non disponibile':
        return np.nan
    elif voto_orale == 'Non classificato': 
        return 0.0

list_VOTI_NAN = ['Non disponibile', 'Senza voto scritto']
def voto_scritto_decode(voto_scritto: str) -> float:
    if voto_scritto in list_VOTI_NUMERICI:
        return float(voto_scritto)
    elif voto_scritto in list_VOTI_NAN:
        return np.nan
    elif voto_scritto == 'Non classificato': 
        return 0.0

def sesso_to_num(sesso: str) -> int:
    return 0 if sesso == "Maschio" else 1

#Features categorica -> One Hot Encoding
list_PROFESSIONI = ['1. Disoccupato/a', '2. Casalingo/a', '3. Dirigente, docente universitario, funzionario o ufficiale militare', '4. Imprenditore/proprietario agricolo', '5. Professionista dipendente, sottuff. militare o libero profession. (medico, av', '6. Lavoratore in proprio (commerciante, coltivatore diretto, artigiano, meccanic', '7. Insegnante, impiegato, militare graduato', '8. Operaio, addetto ai servizi/socio di cooperativa', '9. Pensionato/a', '10. Non disponibile']
PROFESSIONI = from_categorical_to_one_hot_int(list_PROFESSIONI)

#Features categorica -> One Hot Encoding
list_TITOLI = ['1. Licenza elementare', '2. Licenza media', '3. Qualifica professionale triennale', '4. Diploma di maturità', '5. Altro titolo di studio superiore al diploma (I.S.E.F., Accademia di Belle Art', '6. Laurea o titolo superiore (ad esempio Dottorato di Ricerca)', '7. Non disponibile']
TITOLI = from_categorical_to_one_hot_int(list_TITOLI)

#Features categorica -> One Hot Encoding
list_LUOGHI_GENITORI = ['Italia (o Repubblica di San Marino)', 'Unione Europea', 'Paese europeo Non UE', 'Altro', 'Non disponibile']
LUOGHI_GENITORI = from_categorical_to_one_hot_int(list_LUOGHI_GENITORI)

#Features categorica -> One Hot Encoding
list_FREQUENZA_SCUOLA = ['No', 'Sì', 'Non disponibile']
FREQUENZA_SCUOLA = from_categorical_to_one_hot_int(list_FREQUENZA_SCUOLA)

#Features categorica -> One Hot Encoding
list_LUOGO_DI_NASCITA = ['Italia (o Repubblica di San Marino)', 'Unione Europea', 'Paese europeo Non UE', 'Altro', 'Non disponibile']
LUOGO_DI_NASCITA = from_categorical_to_one_hot_int(list_LUOGO_DI_NASCITA)
COLUMN_CONVERTERS = {
    "CODICE_SCUOLA": str, #identificativo della scuola (non considerato)
    "CODICE_PLESSO": str, #identificativo del plesso (non considerato)
    "CODICE_CLASSE": str, #identificato della classe (non considerato)
    "macrotipologia": str, #categoria di scuola (non considerato)
    "campione": int, #campione di riferimento (non considerato)
    "livello": int, # (non considerato)
    "prog": int,
    "CODICE_STUDENTE": str, #codice dello studente (non considerato)
    "sesso": str, #sesso dello studente
    "mese": str, #mese di nascita
    "anno": str, #anno di nascita
    "luogo": str,
    "eta": str, # cosa vuol dire eta?
    "codice_orario": lambda _: np.nan, # unico dato: Mancante di sistema
    "freq_asilo_nido": str,
    "freq_scuola_materna": str,
    "luogo_padre": str,
    "titolo_padre": str,
    "prof_padre": str,
    "luogo_madre": str,
    "titolo_madre": str,
    "prof_madre": str,
    "voto_scritto_ita": lambda voto: voto_scritto_decode(voto),
    "voto_orale_ita": lambda voto: voto_orale_decode(voto),
    "voto_scritto_mat": lambda voto: voto_scritto_decode(voto),
    "voto_orale_mat": lambda voto: voto_orale_decode(voto),
    "D1": lambda result: convert_question_result(result),
    "D2": lambda result: convert_question_result(result),
    "D3_a": lambda result: convert_question_result(result),
    "D3_b": lambda result: convert_question_result(result),
    "D4_a": lambda result: convert_question_result(result),
    "D4_b": lambda result: convert_question_result(result),
    "D4_c": lambda result: convert_question_result(result),
    "D4_d": lambda result: convert_question_result(result),
    "D5_a": lambda result: convert_question_result(result),
    "D5_b": lambda result: convert_question_result(result),
    "D6": lambda result: convert_question_result(result),
    "D7_a": lambda result: convert_question_result(result),
    "D7_b": lambda result: convert_question_result(result),
    "D8": lambda result: convert_question_result(result),
    "D9": lambda result: convert_question_result(result),
    "D10_a": lambda result: convert_question_result(result),
    "D10_b1": lambda result: convert_question_result(result),
    "D10_b2": lambda result: convert_question_result(result),
    "D10_b3": lambda result: convert_question_result(result),
    "D11_a": lambda result: convert_question_result(result),
    "D11_b": lambda result: convert_question_result(result),
    "D12_a": lambda result: convert_question_result(result),
    "D12_b": lambda result: convert_question_result(result),
    "D13_a": lambda result: convert_question_result(result),
    "D13_b": lambda result: convert_question_result(result),
    "D13_c": lambda result: convert_question_result(result),
    "D14": lambda result: convert_question_result(result),
    "D15": lambda result: convert_question_result(result),
    "D16_a": lambda result: convert_question_result(result),
    "D16_b": lambda result: convert_question_result(result),
    "D16_c": lambda result: convert_question_result(result),
    "D16_d": lambda result: convert_question_result(result),
    "D17_a": lambda result: convert_question_result(result),
    "D17_b": lambda result: convert_question_result(result),
    "D18": lambda result: convert_question_result(result),
    "D19_a": lambda result: convert_question_result(result),
    "D19_b": lambda result: convert_question_result(result),
    "D20": lambda result: convert_question_result(result),
    "D21": lambda result: convert_question_result(result),
    "D22": lambda result: convert_question_result(result),
    "D23_a": lambda result: convert_question_result(result),
    "D23_b": lambda result: convert_question_result(result),
    "D23_c": lambda result: convert_question_result(result),
    "D23_d": lambda result: convert_question_result(result),
    "D24_a": lambda result: convert_question_result(result),
    "D24_b": lambda result: convert_question_result(result),
    "D25": lambda result: convert_question_result(result),
    "D26_a": lambda result: convert_question_result(result),
    "D26_b": lambda result: convert_question_result(result),
    "D26_c": lambda result: convert_question_result(result),
    "D26_d": lambda result: convert_question_result(result),
    "regolarità": str,
    "cittadinanza": str,
    "cod_provincia_ISTAT": str,
    "sigla_provincia_istat": str,
    "Nome_reg": str,
    "Cod_reg": str,
    "Areageo_3": str,
    "Areageo_4": str,
    "Areageo_5": str,
    "Areageo_5_Istat": str,
    "Pon": lambda pon: True if pon == "Area_Pon" else False, # lo studente appartiene all'aera Pon oppure no
    "pu_ma_gr": int,
    "pu_ma_no": float,
    "Fattore_correzione_new": float,
    "Cheating": float,
    "PesoClasse": lambda val: float(val) if val != "" else np.nan, # (non considerato)
    "PesoScuola": lambda val: float(val) if val != "" else np.nan, # (non considerato)
    "PesoTotale_Matematica": lambda val: float(val) if val != "" else np.nan, # (non considerato)
    "WLE_MAT": float,
    "WLE_MAT_200": float,
    "WLE_MAT_200_CORR": float,
    "pu_ma_no_corr": float,
    "n_stud_prev": lambda val: int(float(val)),
    "n_classi_prev": lambda val: int(float(val)),
    "LIVELLI": int,
    "DROPOUT": eval
}