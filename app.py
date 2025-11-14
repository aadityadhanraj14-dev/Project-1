from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import openai
from transformers import pipeline
from PIL import Image
import io
import sqlite3
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# SQLite setup
conn = sqlite3.connect("moderation_demo.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS moderation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_type TEXT,
    status TEXT,
    confidence REAL,
    categories TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

image_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
app = FastAPI(title="AI Content Moderation Prototype")

class Feedback(BaseModel):
    content_id: int
    feedback: str

async def moderate_text(content: str):
    response = openai.Moderation.create(model="omni-moderation-latest", input=content)
    flagged = response["results"][0]["flagged"]
    categories = response["results"][0]["categories"]
    confidence = sum(response["results"][0]["category_scores"].values()) / len(response["results"][0]["category_scores"])
    return flagged, confidence, categories

async def moderate_image(file):
    image = Image.open(io.BytesIO(await file.read()))
    predictions = image_classifier(image)
    nsfw_score = next((p['score'] for p in predictions if p['label'].lower() == 'nsfw'), 0)
    flagged = nsfw_score > 0.7
    return flagged, nsfw_score, predictions

def decide(text_flagged, text_conf, img_flagged, img_conf):
    if text_flagged or img_flagged:
        if text_conf > 0.8 or img_conf > 0.8:
            return "BLOCK"
        else:
            return "REVIEW"
    return "SAFE"

@app.post("/moderate-text")
async def moderate_text_endpoint(content: str = Form(...)):
    flagged, confidence, categories = await moderate_text(content)
    status = "BLOCK" if flagged else "SAFE"
    cursor.execute("INSERT INTO moderation_logs (content_type, status, confidence, categories) VALUES (?, ?, ?, ?)",
                   ('text', status, confidence, str(categories)))
    conn.commit()
    return {"status": status, "confidence": confidence, "categories": categories}

@app.post("/moderate-image")
async def moderate_image_endpoint(file: UploadFile):
    flagged, confidence, predictions = await moderate_image(file)
    status = "BLOCK" if flagged else "SAFE"
    cursor.execute("INSERT INTO moderation_logs (content_type, status, confidence, categories) VALUES (?, ?, ?, ?)",
                   ('image', status, confidence, str(predictions)))
    conn.commit()
    return {"status": status, "confidence": confidence, "predictions": predictions}

@app.post("/moderate-combined")
async def moderate_combined(content: str = Form(...), file: UploadFile = None):
    text_flagged, text_conf, text_cat = await moderate_text(content)
    img_flagged, img_conf, img_cat = (False, 0, [])
    if file:
        img_flagged, img_conf, img_cat = await moderate_image(file)
    decision = decide(text_flagged, text_conf, img_flagged, img_conf)
    cursor.execute("INSERT INTO moderation_logs (content_type, status, confidence, categories) VALUES (?, ?, ?, ?)",
                   ('combined', decision, max(text_conf, img_conf), str({"text": text_cat, "image": img_cat})))
    conn.commit()
    return {"decision": decision, "text_confidence": text_conf, "image_confidence": img_conf}

@app.post("/review-feedback")
async def review_feedback(feedback: Feedback):
    cursor.execute("UPDATE moderation_logs SET categories = categories || ' | Feedback: ' || ? WHERE id = ?",
                   (feedback.feedback, feedback.content_id))
    conn.commit()
    return {"message": "Feedback recorded"}
