"""
Descarga el dataset "Hotel Booking Demand" desde Kaggle y lo coloca
en src/data_sample/ para que el proyecto pueda usarlo en local.

Requisitos previos (una sola vez por persona):
1. Tener cuenta en Kaggle.
2. Generar un token API en https://www.kaggle.com/settings -> "Create New Token".
3. Colocar ese archivo en:
   - Windows: C:\\Users\\<tu_usuario>\\.kaggle\\kaggle.json
   - Mac/Linux: ~/.kaggle/kaggle.json

Uso:
    python download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

DATASET_SLUG = "jessemostipak/hotel-booking-demand"
DEST_DIR = Path("src/data_sample")


def main():
    print(f"Descargando dataset '{DATASET_SLUG}' desde Kaggle...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Dataset descargado en cache: {cache_path}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .csv en {cache_path}"
        )

    for csv_file in csv_files:
        destino = DEST_DIR / csv_file.name
        shutil.copy2(csv_file, destino)
        print(f"Copiado -> {destino}")

    print("\nListo. Archivos disponibles en:", DEST_DIR.resolve())


if __name__ == "__main__":
    main()