from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Paths
MODEL_PATH = r"D:\API\xlmr_depression_classifier.pt"
MODEL_NAME = "xlm-roberta-base"

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

# Load model (same architecture used in training)
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# Load trained weights
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)   # strict=False ignores name mismatches
model.eval()

# FastAPI app
app = FastAPI()

# Request format
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "Depressed" if pred == 1 else "Non-Depressed"
    return { "prediction": label}
