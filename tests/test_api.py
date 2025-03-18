import io
import json
from fastapi import UploadFile
import pytest
import torch
from PIL import Image
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app, convert_tabular_to_tensor, load_image_from_uploadfile
from shared.config import TABULAR_DATA_COLUMNS

client = TestClient(app)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_device():
    response = client.get("/device")
    assert response.status_code == 200
    assert response.json() in {"cpu", "cuda"}

def test_convert_tabular_to_tensor_valid():
    tabular_dict = {col: 1.0 for col in TABULAR_DATA_COLUMNS}
    tabular_json = json.dumps(tabular_dict)

    tensor = convert_tabular_to_tensor(tabular_json)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, len(TABULAR_DATA_COLUMNS))

def test_convert_tabular_to_tensor_missing_column():
    # Leave out a required column
    tabular_dict = {col: 1.0 for col in TABULAR_DATA_COLUMNS[:-1]}
    tabular_json = json.dumps(tabular_dict)

    try:
        convert_tabular_to_tensor(tabular_json)
    except Exception as e:
        assert "Missing columns" in str(e.detail)

def test_convert_tabular_to_tensor_extra_column():
    # Add an extra unexpected column
    tabular_dict = {col: 1.0 for col in TABULAR_DATA_COLUMNS}
    tabular_dict["extra"] = 123
    tabular_json = json.dumps(tabular_dict)

    try:
        convert_tabular_to_tensor(tabular_json)
    except Exception as e:
        assert "Unexpected columns" in str(e.detail)

def test_convert_tabular_to_tensor_invalid_type():
    tabular_dict = {col: 1.0 for col in TABULAR_DATA_COLUMNS}
    tabular_dict[TABULAR_DATA_COLUMNS[0]] = "string_val"
    tabular_json = json.dumps(tabular_dict)

    try:
        convert_tabular_to_tensor(tabular_json)
    except Exception as e:
        assert "Invalid value" in str(e.detail)

@patch("app.main.tabular_model")
def test_predict_tabular(mock_model):
    # Mock the model return
    mock_model.return_value = torch.tensor([[0.5]])

    tabular_dict = {col: 1.0 for col in TABULAR_DATA_COLUMNS}
    tabular_json = json.dumps(tabular_dict)

    response = client.post("/predict-tabular", data={"tabular": tabular_json})
    assert response.status_code == 200
    assert "probabilities" in response.json()
    assert "pneumonia" in response.json()["probabilities"]

@pytest.mark.asyncio
async def test_load_image_from_uploadfile_with_valid_image():
    # Create a dummy RGB image (224x224)
    img = Image.new('RGB', (224, 224), color='red')
    
    # Save it to a BytesIO object
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Wrap the BytesIO object in a Starlette UploadFile
    upload_file = StarletteUploadFile(filename="test.jpg", file=img_bytes)

    # Call the function
    loaded_image = await load_image_from_uploadfile(upload_file)

    # Assert it's a valid PIL image in RGB mode
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.mode == "RGB"
    assert loaded_image.size == (224, 224)

@pytest.mark.asyncio
async def test_load_image_from_uploadfile_with_invalid_file():
    fake_file = io.BytesIO(b"This is not an image!")
    upload = UploadFile(filename="fake.txt", file=fake_file)

    with pytest.raises(Exception) as exc_info:
        await load_image_from_uploadfile(upload)

    assert "not a valid image" in str(exc_info.value.detail)

@patch("app.main.img_model")
def test_predict_image(mock_model):
    # Mock the model return
    mock_model.return_value = torch.tensor([[0.5]])

    # Create a dummy image in-memory
    from PIL import Image
    image = Image.new('RGB', (224, 224))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    response = client.post("/predict-image", files=files)

    assert response.status_code == 200
    assert "probabilities" in response.json()
    assert "pneumonia" in response.json()["probabilities"]

@patch("app.main.concat_model")
def test_predict_combined(mock_model):
    # Mock the model return
    mock_model.return_value = torch.tensor([[0.5]])

    # Create a dummy image in-memory
    from PIL import Image
    image = Image.new('RGB', (224, 224))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Create dummy tabular data
    tabular_dict = {col: 1.0 for col in TABULAR_DATA_COLUMNS}
    tabular_json = json.dumps(tabular_dict)

    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'tabular': tabular_json}

    response = client.post("/predict-combined", files=files, data=data)

    assert response.status_code == 200
    assert "probabilities" in response.json()
    assert "pneumonia" in response.json()["probabilities"]