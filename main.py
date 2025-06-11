from flask import Flask
from pywebio.platform.flask import webio_view
from pywebio.input import file_upload
from pywebio.output import put_image, put_markdown
from io import BytesIO
from PIL import Image
from analyzer import FoodAnalyzer

analyzer = FoodAnalyzer(device="mps")
app = Flask(__name__)

def main():
    """
    Dev.Eats
    """
    put_markdown(f"# Welcome to `Dev.Eats`!", position=0)

    while True:
        img_data = file_upload(
            label="Add an image of your food, and I'll tell you what it is!",
            accept="image/*",
        )
        
        image = Image.open(BytesIO(img_data['content']))
        _, result = analyzer.predict(image)

        if img_data:
            put_image(img_data['content'], width="300px")
            put_markdown(f"### {result}")

app.add_url_rule(
    '/', 'webio_view', webio_view(main), 
    methods=['GET', 'POST', 'OPTIONS']
)

if __name__ == '__main__':
    app.run()