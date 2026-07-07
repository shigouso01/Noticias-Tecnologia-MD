from pathlib import Path

carpeta = Path("data/docs")
corpus = []

for archivo in sorted(carpeta.glob("*.txt")):
    with open(archivo, encoding="utf-8") as f:
        corpus.append(f.read())

print(f"Se cargaron {len(corpus)} documentos")