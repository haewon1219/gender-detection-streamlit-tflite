import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Streamlit 앱 구성
st.title("Gender Recognition using Teachable Machine Model")
st.write("Upload an image or use the camera to detect gender.")

# TFLite 모델 경로
tflite_model_path = 'f_m_keras_model.h5.tflite'
labels_path = 'f_m_labels.txt'

# TFLite 모델 로드 및 준비
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model(tflite_model_path)

# 클래스 이름 로드
def load_labels(label_path):
    return open(label_path, "r").readlines()

class_names = load_labels(labels_path)

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.convert('RGB').resize((224, 224))
    img_array = np.array(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape(1, 224, 224, 3)

# 예측 함수
def predict(interpreter, input_details, output_details, img_data):
    interpreter.set_tensor(input_details[0]['index'], img_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# 파일 업로드 또는 카메라 입력
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
use_camera = st.checkbox("Use Camera")

if use_camera:
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
else:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# 예측 처리
if 'image' in locals():
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 이미지 전처리
    img_data = preprocess_image(image)

    # 예측 실행
    output_data = predict(interpreter, input_details, output_details, img_data)

    # 가장 높은 확률의 클래스 선택
    predicted_class = np.argmax(output_data)
    predicted_label = class_names[predicted_class].strip()

    st.write("Predicted Gender:", predicted_label)
    st.write("Confidence Scores:", output_data)
