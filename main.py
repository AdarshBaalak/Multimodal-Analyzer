import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image

import torch
from transformers import pipeline
import easyocr



app = FastAPI(title="Multimodal Analyzer")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"  
DEVICE_INDEX = -1 

print("Device set to:", DEVICE)



print("Loading models... this may take a while on first run.")


sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=DEVICE_INDEX,
)


summarizer_model = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=DEVICE_INDEX,
)



topic_model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=DEVICE_INDEX,
)



toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None,
    device=DEVICE_INDEX,
)



emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=DEVICE_INDEX,
)



image_classifier = pipeline(
    "image-classification",
    model="microsoft/resnet-50",
    device=DEVICE_INDEX,
)



ocr_reader = easyocr.Reader(["en"], gpu=False)

print("All models loaded successfully.")




def ml_sentiment(text: str) -> str:
    if not text.strip():
        return "NEUTRAL"
    res = sentiment_model(text)[0]
    return res.get("label", "NEUTRAL")


def ml_summary(text: str) -> str:
    words = text.split()
    if len(words) < 25:
        return text.strip().capitalize()

    out = summarizer_model(
        text,
        max_length=60,
        min_length=25,
        do_sample=False,
    )
    summary = out[0].get("summary_text", "").strip()
    if not summary:
        return text.strip().capitalize()
    return summary[0].upper() + summary[1:]


def ml_topic(text: str) -> str:
    if not text.strip():
        return "unknown"

    candidate_labels = [
        "mental health",
        "emotions",
        "sadness",
        "happiness",
        "complaint",
        "feedback",
        "daily life",
        "stress",
        "relationships",
        "technology",
        "customer support",
    ]

    res = topic_model(text, candidate_labels)
    labels = res.get("labels", [])
    if labels:
        return labels[0]
    return "unknown"


def ml_toxicity(text: str) -> float:
    if not text.strip():
        return 0.0

    results = toxicity_model(text)
    toxicity_score = 0.0


    if isinstance(results, list) and results and isinstance(results[0], dict):
        iterable = results
    elif isinstance(results, list) and results and isinstance(results[0], list):
        iterable = results[0]
    else:
        iterable = []

    for res in iterable:
        label = res.get("label", "").lower()
        score = res.get("score", 0.0)
        if label in ["toxic", "insult", "obscene", "threat", "severe_toxic"]:
            toxicity_score += score

    return round(min(toxicity_score, 1.0), 2)


def ml_emotion(text: str) -> str:
    if not text.strip():
        return "neutral"

    results = emotion_model(text)

    
    if isinstance(results, list) and results and isinstance(results[0], dict):
        top = max(results, key=lambda x: x.get("score", 0.0))
        return top.get("label", "neutral")
    elif isinstance(results, list) and results and isinstance(results[0], list):
        inner = results[0]
        if inner:
            top = max(inner, key=lambda x: x.get("score", 0.0))
            return top.get("label", "neutral")
    return "neutral"


def ml_image_classification(pil_image: Image.Image) -> str:
    preds = image_classifier(pil_image)
    if not preds:
        return "unknown"
    top = preds[0]
    label = top.get("label", "unknown")
    return label.replace("_", " ").title()


def infer_image_emotion_from_labels(pil_image: Image.Image) -> str:
    preds = image_classifier(pil_image)
    if not preds:
        return "neutral"

    label_text = " ".join([p.get("label", "").lower() for p in preds])

    if any(w in label_text for w in ["cry", "sad", "sorrow", "tragedy"]):
        return "sad"
    if any(w in label_text for w in ["smile", "happy", "joy", "cheer"]):
        return "happy"
    if any(w in label_text for w in ["anger", "angry", "rage", "fury"]):
        return "angry"
    if any(w in label_text for w in ["fear", "scared", "terror"]):
        return "fear"

    return "neutral"


def perform_ocr(pil_image: Image.Image) -> str:
    try:
        result = ocr_reader.readtext(pil_image)
        pieces = [r[1] for r in result if len(r) > 1]
        text = " ".join(pieces).strip()
        return text
    except Exception as e:
        print("OCR error:", e)
        return ""



def fusion_logic(
    text: str,
    text_sentiment: str,
    text_emotion: str,
    image_emotion: str,
    text_toxicity: float,
    ocr_toxicity: float,
    topic: str,
) -> str:
    txt = (text or "").lower()
    max_tox = max(text_toxicity, ocr_toxicity)

    # 1) Toxicity
    if max_tox >= 0.5:
        return (
            "Your message seems to contain strong or abusive language. "
            "Please keep things respectful. If you're facing an issue, "
            "try explaining it calmly so we can better understand and help."
        )

    # 2) Mental health / sadness
    if ("sad" in text_emotion.lower() or "sad" in txt or "depress" in txt) and max_tox < 0.5:
        return (
            "I'm really sorry that you're feeling this way. Your feelings are valid, and "
            "you're not alone. If things feel heavy, consider talking with someone you trust "
            "or reaching out to a mental health professional. Taking that step can really help."
        )

    # 3) Positive mood
    if text_sentiment.upper() == "POSITIVE" or "joy" in text_emotion.lower():
        return (
            "That's great to hear! Thanks for sharing your positive experience. "
            "Keeping track of moments like this can really boost your mood over time. 😊"
        )

    # 4) Complaints / negative yet non-toxic
    if "complaint" in topic.lower() or "customer support" in topic.lower():
        return (
            "Thanks for sharing your concern. We appreciate your feedback and will use it "
            "to improve the experience. If you have more details, feel free to explain."
        )

    # 5) General neutral
    return "Thanks for your input. Your message and image have been analyzed and recorded."


# FastAPI Models
class AnalyzeResponse(BaseModel):
    text_sentiment: str
    text_summary: str
    topic_classification: str
    image_classification: str
    ocr_text: str
    text_toxicity_score: float
    ocr_toxicity_score: float
    text_emotion: str
    image_emotion: str
    automated_response: str



@app.get("/")
def root():
    return {"status": "ok", "message": "Multimodal Analyzer backend running"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    text: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print("Image read error:", e)
        raise ValueError("Invalid image file")

    # Text-based analysis
    text_sentiment = ml_sentiment(text)
    text_summary = ml_summary(text)
    topic = ml_topic(text)
    text_tox = ml_toxicity(text)
    text_em = ml_emotion(text)

    # OCR + toxicity on OCR text
    ocr_text = perform_ocr(pil_image)
    ocr_tox = ml_toxicity(ocr_text) if ocr_text else 0.0

    # Image classification + emotion inference
    img_label = ml_image_classification(pil_image)
    img_emotion = infer_image_emotion_from_labels(pil_image)

    # Fusion logic for automated response
    auto_response = fusion_logic(
        text=text,
        text_sentiment=text_sentiment,
        text_emotion=text_em,
        image_emotion=img_emotion,
        text_toxicity=text_tox,
        ocr_toxicity=ocr_tox,
        topic=topic,
    )

    return AnalyzeResponse(
        text_sentiment=text_sentiment,
        text_summary=text_summary,
        topic_classification=topic,
        image_classification=img_label,
        ocr_text=ocr_text or "None",
        text_toxicity_score=text_tox,
        ocr_toxicity_score=ocr_tox,
        text_emotion=text_em,
        image_emotion=img_emotion,
        automated_response=auto_response,
    )
