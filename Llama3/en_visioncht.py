import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import cv2
import gc

# =============================================
# 1. Set the local model path (directory for LlamaV-o1)
# =============================================
local_model_path = "/media/kanengi/0a4bbc0b-5d3f-479e-8bc4-b087704b7b5d/gemma/llama3/LlamaV-o1"

# =============================================
# 2. Quantization configuration and model loading (4-bit example)
# =============================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Load in 4-bit quantization mode
    bnb_4bit_compute_dtype=torch.bfloat16,   # Compute dtype (you may consider torch.float16 as an alternative)
    bnb_4bit_quant_type="nf4",               # Quantization type (nf4)
    bnb_4bit_use_double_quant=True           # Enable double quantization
)

print("Loading LlamaV-o1 model with 4-bit quantization and offload...")
model = MllamaForConditionalGeneration.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",          # Automatically distribute across GPU and CPU
    offload_folder="offload",   # Offload folder (will be created if it doesn't exist)
    quantization_config=bnb_config
)
model.tie_weights()  # Tie model weights

# =============================================
# 3. Load the processor and tokenizer
# =============================================
# Specify trust_remote_code=True to incorporate settings like chat_template.json
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# =============================================
# 4. Function to capture an image from the camera
# =============================================
def capture_image():
    cap = cv2.VideoCapture(2)  # Adjust the camera ID as needed
    if not cap.isOpened():
        raise Exception("Unable to open the camera.")
    print("Camera is active. Press the spacebar to capture, or 'q' to cancel.")
    captured = False
    image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # Capture image on spacebar press
            captured = True
            # Convert BGR to RGB and convert to a PIL image (explicitly in RGB mode)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame).convert("RGB")
            break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return image if captured and image is not None else None

# =============================================
# 5. Function to generate a description of the image content
# =============================================
def generate_image_description(input_image):
    """
    Using the captured image and a fixed prompt, generate a textual description of the image content.
    """
    prompt = "<|image|><|begin_of_text|>Display whether the equipment in this image complies with Japan's PSE standards."
    # Pass the image and prompt as positional arguments to the processor
    inputs = processor(input_image, prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    # Generation parameters are adjusted for this use-case
    output_ids = model.generate(**inputs, max_new_tokens=400, num_beams=1, num_return_sequences=1, temperature=0.8)
    description = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return description

# =============================================
# 6. Main loop (interactive)
# =============================================
def main():
    print("=== Sample: Image Capture and Content Evaluation using LlamaV-o1 ===")
    while True:
        command = input("Press Enter to capture an image, or 'q' to quit: ")
        if command.lower().strip() == "q":
            print("Exiting.")
            break
        try:
            print("→ Activating the camera...")
            captured_image = capture_image()
            if captured_image is None:
                print("No image was captured.")
                continue
            captured_image.save("captured_image.png")
            print("Image saved as 'captured_image.png'.")
            print("→ Evaluating image content...")
            description = generate_image_description(captured_image)
            print("Generated description:")
            print(description)
        except Exception as e:
            print("An error occurred:", e)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
