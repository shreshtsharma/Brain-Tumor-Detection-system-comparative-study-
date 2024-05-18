import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

labels = ['Glioma Tumor','Meningioma Tumor','No Tumor','Pituitary Tumor']
# Load models
@st.cache_resource
def load_models():
    cnn_model = load_model('cnnModel.keras')
    resnet_model = load_model('ResNetModel.keras')
    xception_model = load_model('XceptionModel.keras')
    return cnn_model, resnet_model, xception_model

cnn_model, resnet_model, xception_model = load_models()
st.title("Brain Tumor Detection System")


selected_models = st.multiselect("Select Models", ['CNN model', 'ResNet model', 'Xception model'])

uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
ind=1
if uploaded_files is not None:
    if st.button("Show Result"):
        for uploaded_file in uploaded_files:
            st.warning("Results for image {}".format(ind))
            ind+=1
    # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                image_array = cv2.resize(image_array,(150,150))
                image_array = image_array.reshape(1,150,150,3)

            
                    # Display input image
                    
                st.image(image, caption='Input Image',  width=400)
                    
                # Making predictions for selected models
                for model_name in selected_models:
                    st.markdown('<hr style="border: 2px solid #565656;">', unsafe_allow_html=True)
                    st.header(f"{model_name} Result")
                    if model_name == 'CNN model':
                        predictions = cnn_model.predict(image_array)  
                    elif model_name == 'ResNet model':
                        predictions = resnet_model.predict(image_array)
                    elif model_name == 'Xception model':
                        predictions = xception_model.predict(image_array)
                # Reshape and preprocess the image


                    indices = predictions.argmax()
                    confidence = predictions[0][indices]
                    if indices!=2: 
                        st.write("Tumor Detected")
                        st.subheader(labels[indices])
                        st.write(f"Probability: {confidence*100:.2f}%")
                    else:
                        st.subheader("No Tumor Detected")
                        st.write(f"Probability: {confidence*100:.2f}%")

            # st.markdown('<hr style="border: 10px solid #314256;">', unsafe_allow_html=True)

            st.title(" ")

