# %% [markdown]
from pathlib import Path
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from preprocessing import cargar_corpus, preprocesar_texto


def cargar_categorias(ruta="data/categorias.csv"):
    categorias = {}
    with open(ruta, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            categorias[fila["documento"]] = fila["categoria"]
    return categorias


if __name__ == "__main__":
    documentos, nombres = cargar_corpus()
    print(f"Se cargaron {len(documentos)} documentos")

    categorias_dict = cargar_categorias()
    y = [categorias_dict[nombre] for nombre in nombres]

    corpus_procesado = [preprocesar_texto(texto) for texto in documentos]
    corpus_para_tfidf = [" ".join(tokens) for tokens in corpus_procesado]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus_para_tfidf)

    chi2_scores, p_values = chi2(X, y)
    feature_names = np.array(vectorizer.get_feature_names_out())

    clases = sorted(set(y))
    print(f"\nClases encontradas: {clases}\n")

    # Top 20 palabras más relevantes por clase
    for clase in clases:
        indices_clase = [i for i, etiqueta in enumerate(y) if etiqueta == clase]
        X_clase = X[indices_clase]
        y_binario = [1 if etiqueta == clase else 0 for etiqueta in y]

        scores_clase, _ = chi2(X, y_binario)
        top_indices = np.argsort(scores_clase)[-20:][::-1]
        top_palabras = feature_names[top_indices]

        print(f"--- {clase} ---")
        print(list(top_palabras))
        print()