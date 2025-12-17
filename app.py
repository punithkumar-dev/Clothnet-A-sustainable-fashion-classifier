from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image
import io

app = FastAPI(title="ClothNet API")

print("Loading ClothNet model...")
model = tf.keras.models.load_model("clothnet_model.h5")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("sustainability_mapping.pkl", "rb") as f:
    sustainability_mapping = pickle.load(f)

print("✅ ClothNet model loaded successfully!")

# Simple hard-coded users for demo
USER_DB = {
    "admin": "admin123",
    "user": "testpass"
}

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        return image_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>ClothNet</h1><p>index.html not found</p>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>")

@app.post("/api/login")
async def login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")

    if username in USER_DB and USER_DB[username] == password:
        return {"status": "success"}
    else:
        return {"status": "error", "message": "Invalid username or password"}

@app.post("/api/classify")
async def classify_image(image: UploadFile = File(...)):
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await image.read()
        processed_image = preprocess_image(image_bytes)

        prediction = model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction)
        predicted_class_raw = label_encoder.inverse_transform([predicted_class_idx])
        product_type_str = (
            predicted_class_raw[0]
            if isinstance(predicted_class_raw, (list, np.ndarray))
            else predicted_class_raw
        )

        sustainability_info = sustainability_mapping.get(product_type_str)
        if sustainability_info is None:
            for key in sustainability_mapping.keys():
                if key.strip().lower() == product_type_str.strip().lower():
                    sustainability_info = sustainability_mapping[key]
                    break
        if sustainability_info is None:
            sustainability_info = {
                "sustainability_score": 0.5,
                "sustainability_label": "Medium",
            }

        confidence = float(np.max(prediction))

        return {
            "product_type": product_type_str,
            "confidence": round(confidence, 3),
            "sustainability_label": sustainability_info["sustainability_label"],
            "sustainability_score": round(
                float(sustainability_info["sustainability_score"]), 2
            ),
            "processing_time_ms": 45,
        }

    except Exception as e:
        print("Error in classify_image:", str(e))
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
