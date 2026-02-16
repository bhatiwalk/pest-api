from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io, base64

app = FastAPI()

model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))

    # Run inference
    results = model(img)

    detections = []

    # Extract detections
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })

    # Draw bounding boxes on image
    processed = results[0].plot()  # numpy array (BGR)

    # Convert to PIL for saving
    processed_pil = Image.fromarray(processed)

    # Convert to Base64
    buffer = io.BytesIO()
    processed_pil.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()

    # RETURN image + detections
    return JSONResponse({
        "image": base64_image,
        "detections": detections
    })

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


#uvicorn main:app --reload --host 0.0.0.0 --port 8000

