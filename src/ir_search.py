# %% [markdown]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from preprocessing import cargar_corpus, preprocesar_texto


def buscar_tfidf(consulta, corpus_para_tfidf, vectorizer, X, nombres, top_n=5):
    consulta_procesada = " ".join(preprocesar_texto(consulta))
    vector_consulta = vectorizer.transform([consulta_procesada])
    similitudes = cosine_similarity(vector_consulta, X).flatten()
    top_indices = np.argsort(similitudes)[::-1][:top_n]
    return [(nombres[i], similitudes[i]) for i in top_indices]


def buscar_bm25(consulta, bm25, corpus_tokenizado, nombres, top_n=5):
    consulta_tokens = preprocesar_texto(consulta)
    scores = bm25.get_scores(consulta_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(nombres[i], scores[i]) for i in top_indices]


def buscar_embeddings(consulta, modelo, embeddings_corpus, nombres, top_n=5):
    vector_consulta = modelo.encode([consulta])
    similitudes = cosine_similarity(vector_consulta, embeddings_corpus).flatten()
    top_indices = np.argsort(similitudes)[::-1][:top_n]
    return [(nombres[i], similitudes[i]) for i in top_indices]


if __name__ == "__main__":
    documentos, nombres = cargar_corpus()
    print(f"Se cargaron {len(documentos)} documentos")

    # --- Preparar TF-IDF ---
    corpus_procesado = [preprocesar_texto(texto) for texto in documentos]
    corpus_para_tfidf = [" ".join(tokens) for tokens in corpus_procesado]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus_para_tfidf)

    # --- Preparar BM25 ---
    bm25 = BM25Okapi(corpus_procesado)  # ya viene tokenizado desde preprocesar_texto

    # --- Preparar Embeddings ---
    modelo_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings_corpus = modelo_embeddings.encode(documentos)

    # --- Consultas de prueba (adaptadas a tu dominio de tecnología) ---
    consultas = [
        "avances en inteligencia artificial",
        "actualización de software",
        "seguridad y ciberataques",
    ]

    for consulta in consultas:
        print(f"\n=== Consulta: '{consulta}' ===")

        print("\n-- TF-IDF + Similitud del coseno --")
        for nombre, score in buscar_tfidf(consulta, corpus_para_tfidf, vectorizer, X, nombres):
            print(f"  {nombre}  (score: {score:.4f})")

        print("\n-- BM25 --")
        for nombre, score in buscar_bm25(consulta, bm25, corpus_procesado, nombres):
            print(f"  {nombre}  (score: {score:.4f})")

        print("\n-- Embeddings semánticos --")
        for nombre, score in buscar_embeddings(consulta, modelo_embeddings, embeddings_corpus, nombres):
            print(f"  {nombre}  (score: {score:.4f})")