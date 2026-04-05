## script that downloads data and saves it to data/raw_data/* both locally and on modal
import gzip
import shutil
import urllib.request
from pathlib import Path

from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image

DATASETS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
]


@app.function(image=build_image(), volumes=VOLUME_MOUNTS)
def download_data(url: str) -> Path:
    filename = url.rsplit("/", maxsplit=1)[-1].removesuffix(".gz")
    output_path = DATA_PATH / "raw_data" / filename
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if output_path.exists():
        print(f"{output_path} is already downloaded")
        return output_path

    print(f"downloading {url}")

    with urllib.request.urlopen(url) as response:
        if url.endswith(".gz"):
            with gzip.GzipFile(fileobj=response) as gz_file, output_path.open("wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)
        else:
            with output_path.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)

    print(f"saved to {output_path}")

    return output_path


@app.local_entrypoint()
def modal_main() -> None: # runs when you call `uv run modal run scripts/download_datasets.py`
    print("Downloading data on modal")
    #for dataset in DATASETS: # version that runs sequentially
    #    path = download_data.remote(dataset)
    #    print(path)
    for path in download_data.map(DATASETS): # version that runs in parallel
        print(path)


if __name__ == "__main__": # runs when you call `uv run scripts/download_datasets.py`
    print("Downloading data locally")
    for dataset in DATASETS:
        download_data.local(dataset)
