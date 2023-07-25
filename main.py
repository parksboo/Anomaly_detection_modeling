from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
from starlette.responses import HTMLResponse
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import numpy as np
import io

app = FastAPI()
model = models.vgg16()
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
model.load_state_dict(torch.load(
    '../bottle_AD.pt', map_location=torch.device('mps')))

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
    pred = model(transformed_img.unsqueeze(0))
    result = np.argmax(pred.detach().numpy())
    if int(result) == 0:
        ans = '비정상'
    else:
        ans = '정상'
    return {"result": ans}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
  <html>
  <body>

  <h2>bottle Anomaly Detection</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
      Select image to upload:
        <input type="file" name="file" id="fileToUpload">
        <input type="submit" value="Upload Image" name="submit">
    </form>

  </body>
  </html>
"""


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
