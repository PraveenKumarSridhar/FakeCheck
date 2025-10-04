import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os

model = tf.keras.models.load_model('vgg19_retrained.h5')
labels = {0: 'fake', 1: 'real'}


def prepare_image(img):
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    predicted_class = np.where(answer > 0.5, 1, 0)
    res = labels[predicted_class[0][0]]
    return res.capitalize()


def run():
    st.title(":male-detective: FakeCheck : Detecting Fake Human Face Images :performing_arts:")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        try:
            img = Image.open(img_file).convert('RGB')
            st.image(img, use_column_width=False)

            # Use a safe directory and ensure it exists
            save_dir = './data/upload_images/'
            os.makedirs(save_dir, exist_ok=True)

            # Sanitize filename to prevent path traversal
            filename = os.path.basename(img_file.name)
            save_image_path = os.path.join(save_dir, filename)

            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())

            result = prepare_image(img)
            print(result)
            st.success("**Predicted : " + result + '**')
        except Exception as e:
            st.error("Error processing image. Please upload a valid image file.")


run()
