import gradio as gr
import numpy as np
import keras.utils as im
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from PIL import Image

# Load the pre-trained model
model = load_model("Tea-LeavesDisease-Detection-Model.h5")

def predict(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    x = im.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    output = np.argmax(model.predict(img_data), axis=1)
    
    index = ['Anthracnose', 'Algal Leaf', 'Bird Eye Spot', 'Brown Blight', 'Gray Light', 'Healthy', 'Red Leaf Spot', 'White Spot']
    result = index[output[0]]
    return result

def main():
    def model_prediction(image):
        prediction = predict(image)
        return prediction
    
    css = """
    body {
        background: #AFDDE5;
        font-family: 'Arial', sans-serif;
    }
    .gradio-container {
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1, h3 {
        text-align: center;
        color: #003135;
    }
    .gradio-container .gr-image-upload, .gradio-container .gr-textbox {
        background: #024950;
        color: white;
    }
    .gradio-container .gr-button {
        background: #0FAA4F;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .gradio-container .gr-button:hover {
        background: #964734;
    }
    .footer {
        text-align: center;
        padding: 10px;
        color: #003135;
        font-size: 12px;
    }
    """
    
    with gr.Blocks(css=css) as interface:
        gr.Markdown("<h1>Tea Leaf Disease Detection System</h1>")
        gr.Markdown("<h3>Upload a tea leaf image and the model will predict the disease.</h3>")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                submit_button = gr.Button("Predict")
            with gr.Column():
                result_output = gr.Textbox(label="Prediction Results")
        
        submit_button.click(fn=model_prediction, inputs=image_input, outputs=result_output)
        
        gr.Markdown("<div class='footer'>Powered by Gradio</div>")
    
    interface.launch(share=True)

if __name__ == "__main__":
    main()
