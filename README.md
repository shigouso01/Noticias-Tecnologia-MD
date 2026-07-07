# Proyecto de Minería de Texto — Instalación de Dependencias

## Requisitos previos
- Python 3.x instalado (el entorno de desarrollo usado en este proyecto es VS Code).
- No se usa `requirements.txt` en este proyecto; cada integrante instala las dependencias manualmente con los comandos de abajo.

## Dependencias a instalar

### Manipulación numérica
```bash
pip install numpy
```

### Machine Learning clásico (TF-IDF, Chi-cuadrado, K-Means, PCA)
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

### Procesamiento de lenguaje natural y NER
```bash
pip install spacy
python -m spacy download es_core_news_sm
```

### Tokenización y stopwords (opcional — a definir por el equipo si se usa en vez de spaCy)
```bash
pip install nltk
```

### Visualización (gráfico de clustering)
```bash
pip install matplotlib
```

### Jupyter (para redactar el reporte final y exportarlo a PDF/HTML)
```bash
pip install jupyter
pip install "nbconvert[webpdf]"
```

Comando de exportación del notebook del reporte final:
```bash
jupyter nbconvert --to webpdf reporte_final.ipynb
```


## Estructura del repositorio (referencia)

```
proyecto-mineria-texto/
├── data/
│   └── docs/
│       ├── doc_01.txt
│       └── ... (mínimo 40 archivos)
├── src/
│   ├── preprocessing.py
│   ├── representation.py
│   ├── ir_search.py
│   ├── feature_selection.py
│   ├── ner_extraction.py
│   └── clustering.py
├── notebooks/
│   └── reporte_final.ipynb
├── outputs/
│   ├── entities.csv
│   ├── clusters.png
│   └── reporte_final.pdf
└── README.md
```

**Nota:** de esta estructura, lo confirmado explícitamente por el documento del proyecto es la carpeta `data/docs/`, el archivo `preprocessing.py`, `entities.csv`, `clusters.png` y el reporte final en PDF.