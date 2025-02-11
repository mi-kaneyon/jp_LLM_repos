import sys
import os
import torch
import gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from janus.models import VLChatProcessor

sys.path.append(os.path.abspath("./Janus"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_path="./Janus-Pro-7B"):
    print(f"[INFO] Loading model from local path: {model_path}")

    # Processor & tokenizer
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # 4bit量子化設定
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,   # or torch.bfloat16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_fp32_cpu_offload=True         # ← これが重要
    )

    # device_map と offload_folder
    device_map = "auto"
    offload_folder = "offload"

    try:
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map=device_map,
            offload_folder=offload_folder
        )
        vl_gpt.eval()
        print("[INFO] Model loaded successfully.")
        return vl_gpt, vl_chat_processor, tokenizer

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

def text_inference(vl_gpt, vl_chat_processor, tokenizer, prompt):
    clear_memory()

    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""}
    ]

    # SFTフォーマット
    input_text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(vl_gpt.device)

    with torch.no_grad():
        # 重要: Janus は multi_modality なので .language_model.generate(...) を呼ぶ
        outputs = vl_gpt.language_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    vl_gpt, vl_chat_processor, tokenizer = load_model("./Janus-Pro-7B")
    
    while True:
        prompt = input("User prompt (type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        result = text_inference(vl_gpt, vl_chat_processor, tokenizer, prompt)
        print("\n---- Janus-Pro Assistant ----")
        print(result + "\n")

if __name__ == "__main__":
    main()
