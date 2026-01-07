import PIL.Image as Image
import requests
from io import BytesIO
import pandas as pd

dataset = pd.read_csv('MM-Food-100K/data.csv')
dataset = dataset[:100]

def get_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

for i in range(len(dataset)):
    img = get_image(dataset['image_url'][i])
    img.save(f'downloaded_images/{i}.jpg')

