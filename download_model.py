# Download Model Script
import os
import requests
from tqdm import tqdm

# URL from your auth token (this is a placeholder - use the actual URL)
MODEL_URL = "https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiajNjbjYxNjQ5bzcxMzdjNHBkcDNic29zIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1NjYwMDg0OH19fV19&Signature=WelEyc0WaGYYKKwSITzFMhShYPBhqv4cWPg-gYD8Q8T-VETnXpneIwChvzmIefxkjT2TDWX6i2ly418XVOJWEE4vRSi0stxmW2AoqhVwDrS6Tk05YDSkXYy9RIbkWbiwLFqwk1nUV9Gj7IUkJdh-0uKNuqK3N4TgljWsT6ImSZGG9yUo7A1OLk1DZoEHI4kAmLobxd9TD7fkOr2oHkIG1-ixVffX2v41R%7E1fZ1chxJpgf-wIBIK9sJ1HlwzahjsYgKhCKJHoO4-7a0RlWd-GNpriyDeLs-zAjdXWnZsrrR6nOhhFoKKym716ajDduCDt0FmGy9AGerLLyHTqYVLLzw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1522391818919792"

def download_model(url, save_path):
    """
    Download the model files from the given URL
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size
    file_size = int(response.headers.get('content-length', 0))
    
    # Open the local file to save the download
    filename = os.path.join(save_path, "model.bin")
    with open(filename, 'wb') as f, tqdm(
        desc="Downloading",
        total=file_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

if __name__ == "__main__":
    MODEL_PATH = "downloaded_model"
    download_model(MODEL_URL, MODEL_PATH)
