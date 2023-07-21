from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.requests import Request
from starlette.responses import HTMLResponse
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import numpy as np
import io

app = FastAPI()
DEVICE = torch.device("mps")
model = models.vgg16().to(DEVICE)
model.classifier[6] = nn.Linear(in_features=4096, out_features=2).to(DEVICE)
model.load_state_dict(torch.load('../bottle_AD.pt'))

# Transform 설정
transform_MVtec = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


# 서버호출 python3 -m uvicorn main:app --reload

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 불러오기 및 전처리
    image = Image.open(io.BytesIO(await file.read()))
    # 변환 적용
    transformed_img = transform_MVtec(image)

    # 예측
    pred = model.predict(transformed_img)
    result = np.argmax(pred)

    return {"result": int(result)}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
  <html>
  <body>

  <h2>bottle Anomaly Detection</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
      Select image to upload:
        <input type="file" name="fileToUpload" id="fileToUpload">
        <input type="submit" value="Upload Image" name="submit">
    </form>

  </body>
  </html>
"""


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
