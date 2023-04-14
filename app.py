import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

model = tf.keras.models.load_model('vgg19_retrained.h5')
labels = {0: 'fake', 1: 'real'}

def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    predicted_class = np.where(answer > 0.5, 1, 0)
    # print(predicted_class)
    res = labels[predicted_class[0][0]]
    # print(res)
    return res.capitalize()


def run():
    st.title(":male-detective: FakeCheck : Detecting Fake Human Face Images :performing_arts:")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './data/upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = prepare_image(save_image_path)
            print(result)
            st.success("**Predicted : " + result + '**')

run()
