# %% [markdown]
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

from preprocessing import cargar_corpus, preprocesar_texto


def generar_clusters(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def graficar_clusters(X, labels, nombre_salida="outputs/clusters.png"):
    X_denso = X.toarray() if hasattr(X, "toarray") else X

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_denso)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis")
    plt.title("Clustering de documentos (K-Means)")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")

    Path("outputs").mkdir(exist_ok=True)  # crea la carpeta si no existe
    plt.savefig(nombre_salida)
    plt.show()


if __name__ == "__main__":
    documentos, nombres = cargar_corpus() # usar "../data/docs" cuando se ejecute jupyter notebook
    print(f"Se cargaron {len(documentos)} documentos")

    corpus_procesado = [preprocesar_texto(texto) for texto in documentos]
    corpus_para_tfidf = [" ".join(tokens) for tokens in corpus_procesado]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus_para_tfidf)
    print("Forma de la matriz TF-IDF:", X.shape)

    labels, modelo_kmeans = generar_clusters(X, n_clusters=3)
    print("Etiquetas de cluster asignadas:", labels)

    graficar_clusters(X, labels)
    print("Gráfico guardado en outputs/clusters.png")
    
# %%
