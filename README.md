# Age_Emotion_Detection

This project involves developing a machine learning model to detect a person’s age and emotional state from a voice note. By extracting vocal features, the system analyzes the characteristics of speech to provide accurate predictions for both age and emotion.
## Features  
- Extracts meaningful vocal features such as MFCCs, pitch, and energy for model input.
- Trains regression models to predict age from voice recordings.
- Implements classification models to detect emotional states (e.g., happy, sad, angry).
- Focuses on accurate predictions without GUI development, highlighting functionality.


## Dataset  
-  You can download the dataset [here]([https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://www.kaggle.com/datasets/rohitzaman/gender-age-and-emotion-detection-from-voice).


## Prerequisites  
Ensure the following are installed:  
- Python 3.8+  
- TensorFlow/Keras  
- NumPy  
- Librosa  
- Matplotlib
- Scikit-learn
- Pytorch

Install dependencies using:  
```bash
pip install -r requirements2.txt

# Age Detection using Fine-Tuned CNN (VGGFace)  

This project demonstrates the fine-tuning of a pre-trained Convolutional Neural Network (CNN), specifically **VGGFace**, to perform age detection on the **IMDB-WIKI** dataset.  

## Features  
- Leverages transfer learning to adapt VGGFace for age detection tasks.  
- Prepares the IMDB-WIKI dataset for effective training and testing.  
- Focuses on functionality and accuracy without the use of a GUI.  
- Implements custom layers for model optimization.  

---

## Dataset  
- **IMDB-WIKI Dataset**: The dataset includes labeled images with age annotations, making it ideal for age detection tasks. You can download the dataset [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

---

## Prerequisites  
Ensure the following are installed:  
- Python 3.8+  
- TensorFlow/Keras  
- NumPy  
- OpenCV (for preprocessing)  
- Matplotlib  

Install dependencies using:  
```bash
pip install -r requirements.txt
