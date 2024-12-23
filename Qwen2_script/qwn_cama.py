import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM
import cv2
import gc
import os

# 環境変数の設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 物体検出モデルの読み込み
detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
detection_model.eval()

# 言語モデルとトークナイザーの読み込み
model_name = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# カメラ映像の取得
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("カメラが開けません。カメラ設定を確認してください。")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_description(question):
    # LLMに質問を送信して回答を得る
    input_text = f"Q: {question}\nA:"
    print(f"LLMへの入力: {input_text}")  # デバッグ用
    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=150)
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # メモリの解放
    del model_inputs
    del generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    print(f"LLMの出力: {response}")  # デバッグ用
    return response

def process_frame(frame):
    # 物体検出の処理
    image = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = detection_model(image)

    if len(outputs) == 0:
        print("物体検出モデルから出力がありません。")
        return frame, []

    detected_objects = []
    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score >= 0.3 and label == 1:  # Personラベルの信頼度が30%以上
            box = box.detach().cpu().numpy().astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            detected_objects.append(box)

    print(f"検出された人物の数: {len(detected_objects)}")
    return frame, detected_objects

def main():
    try:
        # 初回起動時に固定の質問をLLMに送信
        startup_question = "カメラが開かれました。このプログラムは何をするものですか？"
        print("初回質問をLLMに送信します。")
        startup_response = generate_description(startup_question)
        print(f"初回質問の回答: {startup_response}")

        questions = [
            "那个人物是谁？",
            "この画像に写っている人物の好きそうなものについて説明してください。",
            "この画像に写っている人物の名前について説明してください。AIでも評価できます"
        ]

        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み取りに失敗しました。")
                break

            frame_with_boxes, detected_objects = process_frame(frame)

            # 検出された人物に対して質問を送信
            if detected_objects:
                print("人物を検出しました。LLMに質問を送信します。")
                for question in questions:
                    description = generate_description(question)
                    print(f"質問: {question}\n回答: {description}")

            # 映像を表示
            cv2.imshow('Live', frame_with_boxes)

            # ユーザー終了操作
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザーにより終了されました。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # モデルのメモリを解放
        del detection_model
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
