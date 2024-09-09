import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import tempfile
import os

# YOLOモデルのロード (相対パスに変更)
try:
    model = YOLO('weights/best.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

def detect(image_data):
    try:
        # 画像データをPILで読み込む
        image = Image.open(BytesIO(image_data))

        # YOLOv8で推論を行う
        results = model([image], stream=True)  # generator of Results objects

        # 結果の処理
        for result in results:
            # 結果を画像ファイルとして保存する方法
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                result.save(save_dir=os.path.dirname(tmp_file.name))  # 保存先のディレクトリを指定
                tmp_filename = tmp_file.name  # ファイル名を取得

        return tmp_filename  # 処理結果の画像ファイルパスを返す

    except Exception as e:
        st.error(f"Error processing image: {e}")
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
