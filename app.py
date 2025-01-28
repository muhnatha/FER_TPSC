import os
import gdown
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Define the custom EnsembleModel class (same as used for training)
class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.efficientnet_b4 = models.efficientnet_b4(weights='DEFAULT')
        self.resnet = models.resnet50(weights='DEFAULT')

        num_classes = 5  # Adjust this according to your problem
        self.efficientnet_b4.classifier[1] = nn.Linear(self.efficientnet_b4.classifier[1].in_features, num_classes)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        out_efficientnet = self.efficientnet_b4(x)
        out_resnet = self.resnet(x)
        out = (out_efficientnet + out_resnet) / 2
        return out

# Download model from Google Drive if not present
model_path = "model.pth"
if not os.path.exists(model_path):
    st.info("Downloading the model. This may take a few minutes...")
    gdrive_url = "https://drive.google.com/uc?id=1Xqnba2TQI2Fw0_EXACfRp4r5cE9KNIJ2"
    gdown.download(gdrive_url, model_path, quiet=False)

# Load the trained model
device = torch.device('cpu')
model = EnsembleModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define preprocessing transformations
transform = A.Compose([
    A.Resize(224, 224),
    A.ToGray(p=1.0),
    A.GaussianBlur(blur_limit=5, sigma_limit=(0.1, 2.0)),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image_np = np.array(image)
    transformed = transform(image=image_np)
    tensor_image = transformed['image'].unsqueeze(0)
    return tensor_image

# Streamlit App
st.title("Facial Expression Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_tensor = preprocess_image(uploaded_file)

        # Make a prediction
        with torch.no_grad():
            output = model(image_tensor)

        _, predicted_class = torch.max(output, 1)

        # Map predicted class to label
        expression_mapping = {
            0: "Angry",
            1: "Happy",
            2: "Neutral",
            3: "Sad"
        }
        label = predicted_class.item()
        expression = expression_mapping.get(label, "Unknown")

        # Map expressions to song lists
        song_mapping = {
            "Angry": ['"Killing in the Name" – Rage Against the Machine', '"Break Stuff" – Limp Bizkit', '"Headstrong" – Trapt'],
            "Happy": ['"Happy" – Pharrell Williams', '"Can’t Stop the Feeling!" – Justin Timberlake', '"Good as Hell" – Lizzo'],
            "Neutral": ['"Clocks" – Coldplay', '"Photograph" – Ed Sheeran', '"Viva La Vida" – Coldplay'],
            "Sad": ['"Someone Like You" – Adele', '"The Night We Met" – Lord Huron', '"Hurt" – Johnny Cash']
        }
        songs = song_mapping.get(expression, [])

        st.success(f"Detected Expression: {expression}")
        st.write("Suggested Songs:")
        for song in songs:
            st.write(f"- {song}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
