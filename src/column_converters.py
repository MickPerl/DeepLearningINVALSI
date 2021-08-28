import numpy as np

def convert_question_result(result: str) -> bool:
    return result == "Corretta" # alternativamente result == "Errata"

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

COLUMN_CONVERTERS = {
    "CODICE_SCUOLA": lambda x: int(float(x)), #identificativo della scuola (non considerato)
    "CODICE_PLESSO": lambda x: int(float(x)), #identificativo del plesso (non considerato)
    "CODICE_CLASSE": lambda x: int(float(x)), #identificato della classe (non considerato)
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
    "sigla_provincia_istat": lambda prov: str(prov) if prov != np.nan else "ND",
    "Nome_reg": str,
    "Cod_reg": str,
    "Areageo_3": str,
    "Areageo_4": str,
    "Areageo_5": str,
    "Areageo_5_Istat": str,
    "Pon": lambda pon: True if pon == "Area_Pon" else False, # lo studente appartiene all'aera Pon oppure no
    "pu_ma_gr": float,
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
    "DROPOUT": lambda val: 1 if val == "True" else 0 # contiene True e False
}