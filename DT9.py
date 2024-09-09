import streamlit as st
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.wsgi import WSGIMiddleware
from ultralytics import YOLO
from PIL import Image
import torch
from io import BytesIO
import requests

# YOLOモデルのロード
model = YOLO(r'C:\Users\ys041\object detection\OB1\weights\best.pt')

app = FastAPI()

# FastAPIエンドポイント
@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        # 受け取った画像を明示的なパスに保存
        with open(r"C:\Users\ys041\object detection\received_image.png", "wb") as f:
            f.write(image_data)
        
        # 保存された画像をPILで読み込む
        image = Image.open(r"C:\Users\ys041\object detection\received_image.png")
        
        # YOLOv8で推論を行う
        results = model([image], stream=True)  # generator of Results objects

        # 結果の処理
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb 
            result.show()  # 画面に表示
            result.save(filename="result.jpg")  # ディスクに保存

    except Exception as e:
        print(f"Error: {str(e)}")  # エラーをログに出力
        return {"error": str(e)}

@st.cache_resource
def get_app():
    return WSGIMiddleware(app)

st_app = get_app()

def main():
    st.title('Image Upload for YOLOv8 Detection')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])
    if uploaded_file is not None:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'image/png')}
        with st.spinner('Processing...'):
            response = requests.post("http://localhost:8000/detect/", files=files)
            if response.status_code == 200:
                st.image("result.jpg", caption='Processed Image', use_column_width=True)
            else:
                st.error(f"Failed to process image: {response.text}")

if __name__ == "__main__":
    main()
