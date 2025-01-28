from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import gdown

# Define the custom EnsembleModel class (same as used for training)
class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        # Load pre-trained EfficientNet-B4 and ResNet50 models
        self.efficientnet_b4 = models.efficientnet_b4(pretrained=True)
        self.resnet = models.resnet50(pretrained=True)

        # Modify the output layers to match the number of classes in your dataset
        num_classes = 5  # Adjust this according to your problem

        # Replace the classification heads with new layers for both models
        self.efficientnet_b4.classifier[1] = nn.Linear(self.efficientnet_b4.classifier[1].in_features, num_classes)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through both models
        out_efficientnet = self.efficientnet_b4(x)
        out_resnet = self.resnet(x)

        # Average the outputs from the two models
        out = (out_efficientnet + out_resnet) / 2

        return out

# Initialize Flask app
app = Flask(__name__)

# Google Drive file ID and download URL
file_id = "1Xqnba2TQI2Fw0_EXACfRp4r5cE9KNIJ2"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Path to save the model
model_path = "faceRecognition.pth"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading the model from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)

# Load the trained model
device = torch.device('cpu')  # Use 'cuda' if you have a GPU available
model = EnsembleModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Define the same preprocessing transformations used during training
transform = A.Compose([
    A.Resize(224, 224),
    A.ToGray(p=1.0),
    A.GaussianBlur(blur_limit=5, sigma_limit=(0.1, 2.0)),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

def preprocess_image(image):
    # Open the image and convert it to RGB
    image = Image.open(image).convert('RGB')
    
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Apply the albumentations transformations
    transformed = transform(image=image_np)
    
    # Get the tensor from the transformation
    tensor_image = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return tensor_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error="No file uploaded. Please upload an image.")

    file = request.files['file']

    try:
        # Preprocess the image
        image_tensor = preprocess_image(file)

        # Make the prediction
        with torch.no_grad():
            output = model(image_tensor)

        # Get the predicted class (assuming single class output)
        _, predicted_class = torch.max(output, 1)

        # Map the predicted class to a label
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

        # Pass the prediction and songs to the template
        return render_template('index.html', prediction=expression, songs=songs)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
