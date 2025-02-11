import sys
import os
import gc
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from janus.models import VLChatProcessor

# Janus リポジトリへのパスを追加（必要に応じて調整）
sys.path.append(os.path.abspath("./Janus"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clear_memory():
    """CPUモードでも不要なオブジェクトを破棄"""
    gc.collect()

def load_model_cpu(model_path="./Janus-Pro-7B"):
    """
    CPUのみでJanus-Pro-7Bをロードする。
    VRAM不足時にGPUを使用しないための最終的な選択肢。
    速度は非常に遅くなる可能性が高い。
    """
    print(f"[INFO] Loading model on CPU from: {model_path}")

    # 「FutureWarning: The image_processor_class argument is deprecated...」への対策として
    # VLChatProcessor.from_pretrained() 内部で使われる AutoImageProcessor 用引数を
    # 置き換えられるかどうかは、janus のコードに依存しますが、
    # ここでは代替策として "use_fast=False" or "use_fast=True" を指定。
    vl_chat_processor = VLChatProcessor.from_pretrained(
        model_path,
        # huggingface_hub_args={"fast_image_processor_class": None},  # 直接指定が可能ならこう書ける
        # ※ しかし、janus 内部実装によるため、本スクリプトだけで修正は難しい場合があります。
    )

    tokenizer = vl_chat_processor.tokenizer

    # 量子化せず、すべてCPUに載せる場合は quantization_config=None でもOK
    # ただし4bit量子化 + CPUはさらに遅い可能性が高い
    quant_config = None

    try:
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map={"": "cpu"}  # 全てCPUに配置
        )
        vl_gpt.eval()
        print("[INFO] Model loaded successfully (CPU mode).")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    return vl_gpt, vl_chat_processor, tokenizer

def text_inference_cpu(vl_gpt, vl_chat_processor, tokenizer, prompt):
    """
    テキストのみの推論 (CPU版)
    """
    clear_memory()

    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""}
    ]

    input_text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format
    )
    inputs = tokenizer(input_text, return_tensors="pt")

    # CPUのみ
    with torch.no_grad():
        # Janusでは言語モデルに対して vl_gpt.language_model.generate(...) を呼び出す
        outputs = vl_gpt.language_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    model_path = "./Janus-Pro-7B"

    vl_gpt, vl_chat_processor, tokenizer = load_model_cpu(model_path)

    while True:
        prompt = input("User prompt (type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        result = text_inference_cpu(vl_gpt, vl_chat_processor, tokenizer, prompt)
        print("\n---- Janus-Pro Assistant (CPU) ----")
        print(result + "\n")

if __name__ == "__main__":
    main()
