import json
import os
import uuid

import matplotlib.pyplot as plt
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import transforms

from utils import ToTensor4Quantum
from utils.model_utils import get_models

app = Flask(__name__)
CORS(app)

# 配置参数
dataset_name = "GarbageDataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qdevice = "lightning.qubit"
diff_method = "adjoint"

# 加载数据集类别信息
with open("GarbageDataset_Class_Info.json", "r") as f:
    class_info = json.load(f)
    class_names = list(class_info.values())


# 图像预处理函数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        ToTensor4Quantum(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度
    return image.to(device)


# 预测函数
def get_prediction(model, image):
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    return probabilities.cpu().numpy()[0].round(8), predicted_class.item()


# 概率分布结果可视化
def plot_probabilities(probabilities, class_labels, image_path, show=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(class_labels))
    ax.bar(x, probabilities)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")
    ax.set_title('Probability Distribution')
    # Plot Probabilities
    for i, prob in enumerate(probabilities):
        ax.text(i, prob + 0.01, f'{prob:.4f}', ha='center', va='bottom')
    # Adjust layout
    fig.tight_layout()
    # Save probability distribution charts
    fig.savefig(image_path, format='png', dpi=200)
    # Show probability distribution charts
    if show:
        plt.show()

    plt.close(fig)


# 图像预测
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    image = Image.open(request.files['image'].stream).convert('RGB')  # TODO:Check
    image = preprocess_image(image)

    model_name = request.form.get('model_name')  # TODO: Check
    model_weights_path = os.path.join(f"../output", model_name, f"{model_name}_model.pth")
    models = get_models(dataset_name=dataset_name, qdevice=qdevice, qdevice_kwargs={}, diff_method=diff_method)
    model = models[model_name]
    model.to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    probabilities, predicted_class = get_prediction(model, image)

    # 可视化概率分布
    image_filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join("static", image_filename)
    plot_probabilities(probabilities, class_names, image_path, show=False)

    # 构建图像文件的URL
    image_url = f"http://localhost:5001/{image_path}"

    response = {
        "model_name": model_name,
        "predicted_class": class_names[predicted_class],
        "probabilities": dict(zip(class_names, probabilities.tolist())),
        "probability_distribution_url": image_url,
    }

    return jsonify(response)


# 静态文件服务
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
