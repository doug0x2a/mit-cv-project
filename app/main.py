from typing import Dict
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from shared.my_models import ImageModel, ConcatModel
from PIL import Image, UnidentifiedImageError
import pandas as pd
import json
import torch
from torchvision import transforms
import joblib
import os
import io

from shared.config import TABULAR_DATA_COLUMNS

# Load the scikit-learn StandardScaler and PyTorch models
base_dir = os.path.dirname(__file__)
scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
scaler = joblib.load(scaler_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tabular Model
tabular_model_path = os.path.join(base_dir, "models", "tabular-model.pt")
tabular_model = torch.load(tabular_model_path, map_location=device)
tabular_model.eval()

# Image Model
img_model_path = os.path.join(base_dir, "models", "img_model_state_dict.pt")
img_model = ImageModel()
img_model.load_state_dict(torch.load(img_model_path, map_location=device))
img_model.to(device)          
img_model.eval()

# Combined Model using Concatenation
concat_model = ConcatModel(img_model, tabular_model)
concat_model_path = os.path.join(base_dir, "models", "concat_model_state_dict.pt")
concat_model.load_state_dict(torch.load(concat_model_path, map_location=device))
concat_model.to(device)
concat_model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # ImageNet normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI()

@app.get("/")
async def home():
    return "Congratulations! Your API is working as expected. Now head over to <your_server>/docs"

@app.get("/ping")
async def ping():
    """
    Check if the API is running
    """
    return {"status": "ok"}

@app.get("/device")
def get_device():
    """
    Get the device (CPU or GPU) that the model is currently running on
    """
    return device.type

def convert_tabular_to_tensor(tabular:str) -> torch.Tensor:
    """
    Validate the tabular data, apply scaling, and convert it into a PyTorch tensor
    """

    try: 
        tabular_dict = json.loads(tabular)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Check if all expected columns are present
    missing_cols = [col for col in TABULAR_DATA_COLUMNS if col not in tabular_dict]
    extra_cols = [col for col in tabular_dict.keys() if col not in TABULAR_DATA_COLUMNS]

    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    
    if extra_cols:
        raise HTTPException(status_code=400, detail=f"Unexpected columns: {extra_cols}")
    
    # check that all values are floats or ints
    for col, val in tabular_dict.items():
        if not isinstance(val, (int, float)):
            raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {val}")
    
    tabular_df = pd.DataFrame([tabular_dict], columns=TABULAR_DATA_COLUMNS)
    tabular_scaled = scaler.transform(tabular_df)
    tabular_tensor = torch.tensor(tabular_scaled).float().to(device)
    return tabular_tensor

@app.post("/predict-tabular")
async def predict_tabular(tabular: str = Form(...)):
    """
    Predict the probability of pneumonia for a patient based only on tabular data.
    - **tabular**: str = Form(...) - JSON string with tabular data
    """
    tabular_tensor = convert_tabular_to_tensor(tabular)

    with torch.no_grad():
        logit = tabular_model(tabular_tensor)
        pneumonia_prob = torch.sigmoid(logit).item()

    return {"probabilities": {"pneumonia": pneumonia_prob}}

async def load_image_from_uploadfile(file: UploadFile) -> Image.Image:
    """
    Reads an uploaded file and converts it to a PIL RGB Image.
    Raises HTTPException if the file is not a valid image.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the probability of pneumonia for a patient based only on a chest x-ray image.
    - **file**: UploadFile = File(...) - A file to upload
    """

    image = await load_image_from_uploadfile(file)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = img_model(image_tensor)
        pneumonia_prob = torch.sigmoid(logit).item()

    return {"probabilities": {"pneumonia": pneumonia_prob}}


@app.post("/predict-combined")
async def predict_combined(file: UploadFile = File(...), tabular: str = Form(...)):
    """
    Predict the probability of pneumonia for a patient based on a chest x-ray image and tabular data.
    - **file**: UploadFile = File(...) - A file to upload
    - **tabular**: str = Form(...) - JSON string with tabular data
    """
    tabular_tensor = convert_tabular_to_tensor(tabular)

    image = await load_image_from_uploadfile(file)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = concat_model(image_tensor, tabular_tensor)
        pneumonia_prob = torch.sigmoid(logit).item()

    return {"probabilities": {"pneumonia": pneumonia_prob}}