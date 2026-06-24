import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

BASE_DIR = Path("/workspace/marcelo/Deepfake/Datasets/CodecFake/data/wavs")

def mover_arquivo(filepath):
    filename = filepath.name
    if '+' not in filename:
        return

    model = filename.rsplit('+', 1)[0]
    rest = filename.rsplit('+', 1)[1]
    patient = rest.split('_')[0]

    dest_dir = BASE_DIR / model / patient
    dest_dir.mkdir(parents=True, exist_ok=True)

    filepath.rename(dest_dir / rest)

def main():
    print("Listando arquivos...")
    arquivos = list(BASE_DIR.glob("*.wav"))
    print(f"Total de arquivos: {len(arquivos)}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(mover_arquivo, arquivos), total=len(arquivos), desc="Organizando"))

    print("Concluído!")

if __name__ == "__main__":
    main()