# Potato Leaf Disease Detection

This project implements a Convolutional Neural Network (CNN) to detect potato leaf diseases (Early Blight, Late Blight) or identify if the leaf is Healthy. It is built using TensorFlow/Keras and deployed as a web application using Streamlit.

## Setup Instructions

1. **Install Dependencies**
   Make sure you have Python installed. Then, run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   Run the training script to build and train the CNN model. The script uses `kagglehub` to **automatically** download the PlantVillage dataset for you!
   ```bash
   python train_model.py
   ```
   This will automatically fetch the dataset, train the model on the Potato classes, generate accuracy/loss plots, and save the trained model as `potato_disease_model.h5`.

3. **Run the Streamlit App**
   Once the model is trained and saved, you can launch the web application:
   ```bash
   streamlit run app.py
   ```
   This will open a browser window where you can upload potato leaf images and get predictions in real-time.
