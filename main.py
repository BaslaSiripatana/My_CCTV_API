from fastai.vision.all import load_learner
from fastapi import FastAPI, HTTPException, UploadFile, File
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import io
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# กำหนดค่า CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ระบุ origin ที่อนุญาต
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตให้ใช้ method ทุกชนิด
    allow_headers=["*"],  # อนุญาตให้มี headers ทุกชนิด
)

# Load the learner object
learn = load_learner('cctv_model2.pth')

# Access the underlying PyTorch model
model = learn.model
model.eval()

# Define image transformation for inference
transform = Compose([lambda x: ToTensor()(x.convert('RGB'))])

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# Endpoint for making predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()

        # Preprocess the input image
        pil_image = Image.open(io.BytesIO(contents))
        input_tensor = transform(pil_image).unsqueeze(0)

        # Make the prediction using the PyTorch model
        with torch.no_grad():
            output = model(input_tensor)

        # Process the output
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probability, predicted_class = torch.max(probabilities, 0)

        # Convert to Python types
        probability = probability.item()
        predicted_class = predicted_class.item()

        # Return prediction result and probability
        return {"prediction": learn.dls.vocab[predicted_class], "probability": probability}

    except Exception as e:
        # Handle errors appropriately
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app)


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), timeout_keep_alive=120, timeout_notify=120)



