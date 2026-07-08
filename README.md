# Proyecto de Minería de Texto — Instalación de Dependencias

## Requisitos previos
- Python 3.x instalado (el entorno de desarrollo usado en este proyecto es VS Code).
- No se usa `requirements.txt` en este proyecto; cada integrante instala las dependencias manualmente con los comandos de abajo.

## Dependencias realmente utilizadas en el proyecto

Esta tabla mapea cada dependencia contra el archivo `.py`, para entender el uso de cada dependencia:

| Dependencia | Usada en | Para qué |
|---|---|---|
| `numpy` | `feature_selection.py`, `ir_search.py` | Manejo de arrays y ordenamiento de resultados (`np.argsort`) |
| `scikit-learn` | `preprocessing.py`, `clustering.py`, `feature_selection.py`, `ir_search.py` | `TfidfVectorizer`, `KMeans`, `PCA`, `chi2`, `cosine_similarity` |
| `sentence-transformers` | `representation.py`, `ir_search.py` | Generación de embeddings semánticos |
| `rank-bm25` | `ir_search.py` | Búsqueda con BM25 |
| `spacy` | `preprocessing.py`, `ner_extraction.py`, `relation_extraction.py` | Tokenización, lematización, NER y extracción de relaciones |
| `matplotlib` | `clustering.py` | Generación de `clusters.png` |
| `jupyter` | `notebooks/reporte_final.ipynb` | Redacción del reporte final |
| `nbconvert` + `playwright` | `notebooks/reporte_final.ipynb` | Exportar el reporte final a PDF |

## Comandos de instalación

### Manipulación numérica
```bash
pip install numpy
```

### Machine Learning clásico (TF-IDF, Chi-cuadrado, K-Means, PCA, similitud del coseno)
```bash
pip install scikit-learn
```

### Embeddings semánticos
```bash
pip install sentence-transformers
```

Si la instalación falla con un error relacionado a `urllib3` / lectura interrumpida durante la descarga, probar:
```bash
pip install --no-cache-dir sentence-transformers
```
o actualizar pip primero:
```bash
pip install --upgrade pip
```

### Búsqueda BM25
```bash
pip install rank-bm25
```

### Procesamiento de lenguaje natural, NER y extracción de relaciones
```bash
pip install spacy
python -m spacy download es_core_news_md
```

**Nota:** el proyecto probó tanto `es_core_news_sm` como `es_core_news_md`. Se confirmó que `es_core_news_md` ofrece mejor cobertura y precisión de entidades (ver sección de Limitaciones del reporte final), por lo que es el modelo recomendado para instalar. `es_core_news_sm` ya no es necesario si se usa `md`.

### Visualización (gráfico de clustering)
```bash
pip install matplotlib
```

### Jupyter (para redactar el reporte final y exportarlo a PDF/HTML)
```bash
pip install jupyter
pip install "nbconvert[webpdf]"
playwright install chromium
```

Comando de exportación del notebook del reporte final:
```bash
jupyter nbconvert --to webpdf reporte_final.ipynb
```

## Estructura del repositorio

```
proyecto-mineria-texto/
├── data/
│   ├── docs/
│   │   ├── 1.txt
│   │   └── ... (40 archivos)
│   └── categorias.csv
├── src/
│   ├── preprocessing.py
│   ├── representation.py
│   ├── ir_search.py
│   ├── feature_selection.py
│   ├── ner_extraction.py
│   ├── clustering.py
│   └── relation_extraction.py
├── notebooks/
│   └── reporte_final.ipynb
├── outputs/
│   ├── entities.csv
│   ├── relations.csv
│   ├── clusters.png
│   └── reporte_final.pdf
└── README.md
```

**Nota:** de esta estructura, lo confirmado explícitamente por el documento del proyecto es la carpeta `data/docs/`, el archivo `preprocessing.py`, `entities.csv`, `clusters.png` y el reporte final en PDF.
