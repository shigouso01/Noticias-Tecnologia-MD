# %% [markdown]
import spacy
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("es_core_news_sm")

def preprocesar_texto(texto):
    doc = nlp(texto.lower())  # minúsculas

    tokens_limpios = []
    for token in doc:
        if token.is_punct:      # eliminar signos de puntuación
            continue
        if token.like_num:      # eliminar números
            continue
        if token.is_stop:       # eliminar stopwords
            continue
        if token.is_space:      # eliminar espacios/saltos de línea sueltos
            continue
        tokens_limpios.append(token.lemma_)  # lematización

    return tokens_limpios


def cargar_corpus(carpeta="data/docs"):
    carpeta = Path(carpeta)
    documentos = []
    nombres = []
    for archivo in sorted(carpeta.glob("*.txt")):
        with open(archivo, encoding="utf-8") as f:
            documentos.append(f.read())
            nombres.append(archivo.name)
    return documentos, nombres


if __name__ == "__main__":
    documentos, nombres = cargar_corpus()
    print(f"Se cargaron {len(documentos)} documentos")

    corpus_procesado = []
    for texto in documentos:
        tokens = preprocesar_texto(texto)
        corpus_procesado.append(tokens)

    # Muestra de verificación: el primer documento antes y después
    print("\n--- Ejemplo (documento 1) ---")
    print("Original:", documentos[0])
    print("Procesado:", corpus_procesado[0])

    # --- Representación vectorial: TF-IDF (Sección V) ---
    corpus_para_tfidf = [" ".join(tokens) for tokens in corpus_procesado]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus_para_tfidf)

    print("\n--- TF-IDF ---")
    print("Forma de la matriz X:", X.shape)  # (n_documentos, n_palabras_únicas)
    print("Primeras 10 palabras del vocabulario:", vectorizer.get_feature_names_out()[:10])