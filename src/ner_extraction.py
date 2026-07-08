# %% [markdown]
import csv
from pathlib import Path
import spacy

from preprocessing import cargar_corpus

nlp = spacy.load("es_core_news_md")  # modelo mediano, más rápido que el grande y suficiente para NER

# Mapeo de etiquetas de spaCy a las categorías que pide el documento
ETIQUETAS_DESEADAS = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "DATE": "DATE",
    "EVENT": "EVENT",
    "PRODUCT": "PRODUCT",
}


def extraer_entidades(texto):
    doc = nlp(texto)
    resultados = []
    for ent in doc.ents:
        tipo = ETIQUETAS_DESEADAS.get(ent.label_)
        if tipo is None:
            continue  # ignorar etiquetas que no pidió el documento (MISC, etc.)
        resultados.append({
            "entidad": ent.text,
            "tipo": tipo,
            "contexto": doc[max(ent.start - 5, 0):min(ent.end + 5, len(doc))].text
        })
    return resultados


if __name__ == "__main__":
    documentos, nombres = cargar_corpus()
    print(f"Se cargaron {len(documentos)} documentos")

    filas = []
    for nombre, texto in zip(nombres, documentos):
        entidades = extraer_entidades(texto)
        for e in entidades:
            filas.append({
                "documento": nombre,
                "entidad": e["entidad"],
                "tipo": e["tipo"],
                "contexto": e["contexto"]
            })

    Path("outputs").mkdir(exist_ok=True)
    ruta_salida = "outputs/entities.csv"

    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["documento", "entidad", "tipo", "contexto"])
        writer.writeheader()
        writer.writerows(filas)

    print(f"Se extrajeron {len(filas)} entidades")
    print(f"Guardado en {ruta_salida}")