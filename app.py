from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from huggingface_hub import hf_hub_download

# Hugging Face repo + file
REPO_ID = "Aroojzahra908/model"   # your HF repo
FILENAME = "xlmr_depression_classifier.pt"
MODEL_NAME = "xlm-roberta-base"

# Download model file from Hugging Face Hub
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

# Load model architecture
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Load trained weights
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# FastAPI app
app = FastAPI()

# Input schema
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
    return {"input": data.text, "prediction": label}
