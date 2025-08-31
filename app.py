# app.py
import os
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image

# Configure API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# Serve static files (edited images)
os.makedirs("rectified_image", exist_ok=True)
app.mount("/rectified_image", StaticFiles(directory="rectified_image"), name="rectified_image")

def load_image_as_base64(file: UploadFile):
    return base64.b64encode(file.file.read()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head><title>Image Editor</title></head>
        <body>
            <h1>Upload an image to edit</h1>
            <form action="/edit-image/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input name="prompt" type="text" placeholder="Enter prompt" size="50">
                <input type="submit" value="Edit Image">
            </form>
        </body>
    </html>
    """

@app.post("/edit-image/")
async def edit_image(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Load image as base64
        image_b64 = load_image_as_base64(file)

        # Step 1: Generate instructions from Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_b64},
                prompt
            ]
        )
        instructions = response.text

        # Step 2: Apply Pix2Pix pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ).to(device)

        image = Image.open(file.file).convert("RGB")
        edited_image = pipe(instructions, image=image, num_inference_steps=30).images[0]

        # Save edited image
        edited_image_path = f"rectified_image/edited_{file.filename}"
        edited_image.save(edited_image_path)

        # Return edited image URL
        return HTMLResponse(f'<h2>Edited Image</h2><img src="/{edited_image_path}" />')
    except Exception as e:
        return {"error": str(e)}
    
#run uvicorn app:app --reload
