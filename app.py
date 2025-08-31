# app_preview_download_fixed.py
import os
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import google.generativeai as genai
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image

# Configure API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# Create folder for edited images
os.makedirs("rectified_image", exist_ok=True)

# Serve static files for download
from fastapi.staticfiles import StaticFiles
app.mount("/rectified_image", StaticFiles(directory="rectified_image"), name="rectified_image")

def load_image_as_base64(file: UploadFile):
    file.file.seek(0)
    return base64.b64encode(file.file.read()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head><title>Image Editor</title></head>
        <body>
            <h1>Upload an image to edit</h1>
            <form action="/edit-image/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" required>
                <input name="prompt" type="text" placeholder="Enter prompt" size="50" required>
                <input type="submit" value="Edit Image">
            </form>
        </body>
    </html>
    """

@app.post("/edit-image/", response_class=HTMLResponse)
async def edit_image(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Load image as base64 for Gemini
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

        # --- Truncate instructions to avoid CLIP 77-token limit ---
        instructions = " ".join(instructions.split()[:75])

        # Step 2: Apply Pix2Pix pipeline
        file.file.seek(0)
        image = Image.open(file.file).convert("RGB")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ).to(device)
        edited_image = pipe(instructions, image=image, num_inference_steps=30).images[0]

        # Save edited image
        edited_image_path = f"rectified_image/edited_{file.filename}"
        edited_image.save(edited_image_path)

        # Convert to base64 for HTML preview
        buffered = BytesIO()
        edited_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return preview with download link
        return HTMLResponse(f"""
        <html>
            <body>
                <h2>Edited Image Preview</h2>
                <img src="data:image/png;base64,{img_str}" style="max-width:80%;"><br><br>
                <a href="/rectified_image/edited_{file.filename}" download>Download Edited Image</a><br><br>
                <a href="/">Edit Another Image</a>
            </body>
        </html>
        """)

    except Exception as e:
        # Friendly HTML error page
        return HTMLResponse(f"""
            <html>
                <body>
                    <h3>Error: {str(e)}</h3>
                    <a href="/">Go back</a>
                </body>
            </html>
        """)
