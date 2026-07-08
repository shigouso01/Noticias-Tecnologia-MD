# %% [markdown]
from sentence_transformers import SentenceTransformer

def generar_embeddings(corpus_textos):
    
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = modelo.encode(corpus_textos)
    return embeddings


if __name__ == "__main__":
    from preprocessing import cargar_corpus

    documentos, nombres = cargar_corpus()
    embeddings = generar_embeddings(documentos)

    print(f"Se generaron embeddings para {len(documentos)} documentos")
    print("Forma de la matriz de embeddings:", embeddings.shape)