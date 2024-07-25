import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM
import cv2
import gc

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 物体検出モデルの読み込み
detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
detection_model.eval()

# 言語モデルの読み込み
model_name = "internlm/internlm2_5-7b-chat"  # ここは現行のモデルに変更
language_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# カメラ映像の取得
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("カメラが開けません。カメラ設定を確認してください。")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 画像の前処理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_description():
    prompt = "この人物の特徴を説明してください。"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            output_ids = language_model.generate(input_ids=inputs.input_ids, max_new_tokens=500, do_sample=True, temperature=0.8)
            description = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            description = f"説明生成中にエラーが発生しました: {str(e)}"
    return description

def process_frame(frame):
    image = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = detection_model(image)

    detected_objects = []

    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score >= 0.3 and label == 1:  # Personラベルの信頼度が30%以上
            box = box.detach().cpu().numpy().astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            detected_objects.append(box)

    return frame, detected_objects

def main():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み取りに失敗しました。")
                break

            frame_with_boxes, detected_objects = process_frame(frame)

            for box in detected_objects:
                description = generate_description()
                print(f"人物検出: {description}")

            cv2.imshow('Live', frame_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザーにより終了されました。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # モデルのメモリを解放
        del detection_model
        del language_model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
