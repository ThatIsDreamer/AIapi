import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import io
import asyncio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

skin_type_model = models.resnet18(pretrained=False)
try:
    state_dict_skin_type = torch.load("skin_type1905 (1).pth", map_location=device)
    skin_type_model.load_state_dict(state_dict_skin_type)
    skin_type_model = skin_type_model.to(device)
    skin_type_model.eval()
    print("Skin type model loaded successfully from /content/skin_type1905 (1).pth.")
except FileNotFoundError:
    print("Error: '/content/skin_type1905 (1).pth' not found. Please ensure the model file is in the correct path.")
    exit()
except Exception as e:
    print(f"Error loading skin type model: {e}")
    exit()

acne_model = models.efficientnet_b0(pretrained=False)

num_acne_classes = 2

in_features = acne_model.classifier[1].in_features
acne_model.classifier[1] = torch.nn.Linear(in_features, num_acne_classes)

try:
    state_dict_acne = torch.load("acne_model.pth", map_location=device)
    acne_model.load_state_dict(state_dict_acne)
    acne_model = acne_model.to(device)
    acne_model.eval()
    print("Acne model (EfficientNet_B0) loaded successfully with adjusted classifier.")
except FileNotFoundError:
    print("Error: 'acne_model.pth' not found at the specified path. Please ensure the model file is there.")
    exit()
except Exception as e:
    print(f"Error loading acne model: {e}. Double-check the number of output classes (should be {num_acne_classes}) and the model's exact architecture.")
    exit()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def resize_with_padding(image: Image.Image, target_size=(214, 214)) -> Image.Image:
    ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_img

def predict_skin_type(model: torch.nn.Module, image_arr: List[Image.Image]) -> str:
    skin_type_predictions = []
    class_names = ["Dry", "Normal", "Oily"]

    for img_pil in image_arr:
        img = resize_with_padding(img_pil, (214, 214))
        img_tensor = transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            if abs(probs[0][0] - probs[0][1]) < 0.15:
                predicted_class = 2
            elif 0.15 < abs(probs[0][0] - probs[0][1]) < 0.3:
                predicted_class = 0
        skin_type_predictions.append(class_names[predicted_class])

    if 'Normal' in skin_type_predictions:
        return 'Normal'
    elif 'Dry' in skin_type_predictions:
        return 'Dry'
    else:
        return 'Oily'

def predict_acne(model: torch.nn.Module, image_arr: List[Image.Image]) -> str:
    acne_predictions = []
    class_names_from_model_output = ["Acne", "No Acne"]
    final_acne_status_names = ["Acne", "No Acne", "Hesitation"]

    for img_pil in image_arr:
        img = resize_with_padding(img_pil, (214, 214))
        img_tensor = transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)

            if probs.shape[1] != num_acne_classes:
                print(f"Warning: Acne model outputted {probs.shape[1]} classes, but expected {num_acne_classes}. "
                      "This might affect 'Hesitation' logic.")

            print(f"Acne probabilities for one image: {probs.tolist()}")

            predicted_class_idx = torch.argmax(probs, dim=1).item()
            predicted_class_name = class_names_from_model_output[predicted_class_idx]

            if len(probs[0]) >= 2 and abs(probs[0][0] - probs[0][1]) < 0.2:
                acne_predictions.append("Hesitation")
            else:
                acne_predictions.append(predicted_class_name)

    if acne_predictions.count("Acne") >= 1:
        return "Acne"
    if acne_predictions.count("Hesitation") >= 1:
        return "Hesitation"
    return "No Acne"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    df = pd.read_csv("products_treatment_annotated.csv")
    print("Product data CSV loaded successfully.")
except FileNotFoundError:
    print("Error: 'products_treatment_annotated.csv' not found. Product filtering will not work.")
    df = pd.DataFrame()
except Exception as e:
    print(f"Error loading product data CSV: {e}")
    df = pd.DataFrame()

@app.post("/predict/", summary="Predict Skin Type and Acne Status from Images")
async def predict(files: List[UploadFile] = File(..., description="List of image files to analyze")):
    images = []
    for f in files:
        try:
            contents = await f.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error processing uploaded file {f.filename}: {e}")
            continue

    if not images:
        return {"error": "No valid images provided for prediction."}

    skin_type = await asyncio.to_thread(predict_skin_type, skin_type_model, images)
    acne_status = await asyncio.to_thread(predict_acne, acne_model, images)

    has_acne = (acne_status == "Acne")

    return {
        "skin_type": skin_type,
        "acne": has_acne,
        "acne_status_detail": acne_status
    }

@app.post("/filter/", summary="Filter Products based on Skin Type and Conditions")
async def filter_products(
    min_price: int = Form(..., description="Minimum price for products"),
    max_price: int = Form(..., description="Maximum price for products"),
    skin_type: str = Form(..., description="Predicted skin type (e.g., 'normal', 'dry', 'oily')"),
    acne: bool = Form(False, description="Filter for products suitable for acne-prone skin"),
    comedones: bool = Form(False, description="Filter for products suitable for comedones"),
    rosacea: bool = Form(False, description="Filter for products suitable for rosacea"),
):
    if df.empty:
        return {"error": "Product data not loaded. Cannot filter products."}

    result = await asyncio.to_thread(filter_products_sync, min_price, max_price, skin_type, acne, comedones, rosacea)
    return result

def filter_products_sync(min_price, max_price, skin_type, acne, comedones, rosacea):
    df_filtered = df[(df["Цена"] >= min_price) & (df["Цена"] <= max_price)]

    if acne:
        df_filtered = df_filtered[df_filtered["Подходит для акне"] == True]
    if comedones:
        df_filtered = df_filtered[df_filtered["Подходит для комедонов"] == True]
    if rosacea:
        df_filtered = df_filtered[df_filtered["Подходит для розацеа"] == True]

    skin_type_keywords = {
        "normal": ["для нормальной кожи", "для комбинированной кожи"],
        "dry": ["для сухой кожи", "для обезвоженной кожи"],
        "oily": ["для жирной кожи"]
    }
    if acne or comedones or rosacea:
        for k in skin_type_keywords:
            skin_type_keywords[k].append("для проблемной кожи")

    keywords = skin_type_keywords.get(skin_type.lower(), [])

    df_filtered = df_filtered[
        df_filtered["Тип кожи"].astype(str).str.contains("all", case=False, na=False) |
        df_filtered["Тип кожи"].astype(str).apply(
            lambda x: any(keyword in x.lower() for keyword in keywords)
        )
    ]

    return df_filtered[["Название", "Цена", "Тип продукта"]].reset_index(drop=True).head(10).to_dict(orient="records")
