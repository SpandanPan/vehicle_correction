import os
import base64
import google.generativeai as genai
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def load_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_path = "image/Car_1.jpg" 
edited_image_path = "rectified_image/car_rectified.png"  
image_b64 = load_image_as_base64(image_path)
prompt = "Please rectify and describe the car image: adjust brightness/contrast, fix perspective issues, and highlight key details."

def main(prompt,image_b64):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
    [
        {"mime_type": "image/jpeg", "data": image_b64},
        prompt
    ]
    instructions = response.text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to(device)
    image = Image.open(image_path).convert("RGB")
    edited_image = pipe(instructions, image=image, num_inference_steps=30).images[0]
    edited_image.save(edited_image_path)
)
    


if __name__ == "__main__":
    main()
