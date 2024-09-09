import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
from io import BytesIO
import tempfile

# YOLOモデルのロード (相対パスに変更)
model = YOLO('weights/best.pt')

def detect(image_data):
    try:
        # 画像データをPILで読み込む
        image = Image.open(BytesIO(image_data))

        # YOLOv8で推論を行う
        results = model([image], stream=True)  # generator of Results objects

        # 結果の処理
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs

            # 結果を一時ファイルに保存
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                result.save(save_dir=tmp_file.name)  # 結果を保存
                tmp_filename = tmp_file.name  # ファイル名を取得

        return tmp_filename  # 処理結果の画像ファイルパスを返す

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    st.title('Image Upload for YOLOv8 Detection')

    # 画像をアップロード
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()

        with st.spinner('Processing...'):
            result_image_path = detect(image_data)

            if result_image_path:
                # 処理結果の画像を表示
                st.image(result_image_path, caption='Processed Image', use_column_width=True)
            else:
                st.error("Failed to process the image.")

if __name__ == "__main__":
    main()

