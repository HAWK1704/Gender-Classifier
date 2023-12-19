from tensorflow.keras.models import load_model
import tensorflow as tf
import gradio as gr
import numpy as np

loaded_model = load_model("gender_classifier_model.h5")


def myfun(img):
    # Gradio automatically converts the input image to a NumPy array
    # Convert the image to the required input format for the model
    img = tf.image.resize(img, (64, 64))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Use the loaded model for predictions
    loaded_classes = loaded_model.predict(x, batch_size=1)
    print(loaded_classes)
    if loaded_classes[0] > 0.5:
        return 'Is a Man'
    else:
        return 'Is A Woman'


iface = gr.Interface(fn=myfun, inputs=gr.Image(label='Drop an Image or Open Camera to Classify'), outputs=gr.Text())
iface.launch()
