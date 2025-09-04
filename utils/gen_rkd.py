import torch
import clip
from PIL import Image
import json
import os
import pickle
from tqdm import tqdm

# --- 1. 配置路径和参数 ---
# 请根据您的实际环境修改这些路径
annotation_file = "/22liushoulong/datasets/ZSFooD2/annotations/instances_train2017.json"
image_directory = "/22liushoulong/datasets/ZSFooD2/train2017/"
output_directory = "/22liushoulong/datasets/ZSFooD2/gt_classagnostic_distilfeats/"

# --- 2. 初始化模型 ---
# 自动选择设备（如果可用，则使用CUDA，否则使用CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载CLIP模型和预处理器
# 'ViT-B/32' 是模型名称，您可以根据需要更改
print("Loading CLIP model 'ViT-B/32'...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded successfully.")

# --- 3. 准备工作 ---
# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 加载标注文件
print(f"Loading annotations from {annotation_file}...")
with open(annotation_file, 'r') as f:
    annotations_data = json.load(f)

# 为了提高效率，将标注按 image_id 进行分组
annotations_map = {}
print("Mapping annotations to image IDs...")
if 'annotations' not in annotations_data:
    raise KeyError("错误：在JSON文件中未找到 'annotations' 键。请确认文件格式是否正确，并包含边界框信息。")

for ann in tqdm(annotations_data['annotations'], desc="Processing Annotations"):
    image_id = ann['image_id']
    if image_id not in annotations_map:
        annotations_map[image_id] = []
    annotations_map[image_id].append(ann)

# --- 4. 主处理循环 ---
print(f"\nStarting feature extraction for {len(annotations_data['images'])} images...")
# 使用 tqdm 显示处理进度
for image_info in tqdm(annotations_data['images'], desc="Extracting Features"):
    image_id = image_info['id']
    file_name = image_info['file_name']
    image_path = os.path.join(image_directory, file_name)

    # 图片级别的错误处理：如果图片文件不存在或无法打开，则跳过整个图片
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found, skipping: {image_path}")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error: Could not open or convert image {image_path}. Skipping. Error: {e}")
        continue

    # 获取该图片对应的所有标注
    image_annotations = annotations_map.get(image_id, [])
    if not image_annotations:
        continue

    # 用于保存该图片所有 gt box 特征的列表
    results_for_image = []

    # 遍历该图片的所有标注
    for ann in image_annotations:
        # =========================================================
        #  ↓↓↓ 全局错误捕获（核心改动） ↓↓↓
        # =========================================================
        # 使用 try-except 块包裹单个标注的处理流程
        # 这样任何一个标注出错，都不会影响其他标注和整个程序的运行
        try:
            if 'bbox' not in ann:
                continue

            box = ann['bbox'] # [x, y, width, height]
            x, y, w, h = box

            # 检查box的宽高是否有效
            if w <= 0 or h <= 0:
                continue

            # 根据box坐标裁剪图片
            cropped_image = image.crop((x, y, x + w, y + h))

            # 使用CLIP的预处理器处理裁剪出的图片
            preprocessed_image = preprocess(cropped_image).unsqueeze(0).to(device)

            # 使用CLIP模型提取特征
            with torch.no_grad():
                image_features = clip_model.encode_image(preprocessed_image)

            # 将结果存入列表
            result_tuple = (box, image_features.cpu())
            results_for_image.append(result_tuple)

        except Exception as e:
            # 如果处理单个标注时发生任何错误，打印警告并跳过该标注
            ann_id = ann.get('id', 'N/A') # 尝试获取标注ID用于调试
            print(f"Warning: Skipped annotation (id: {ann_id}) for image '{file_name}' due to an error: {e}")
            continue # 继续处理下一个标注
        # =========================================================

    # 如果成功提取了至少一个特征，则保存为pkl文件
    if results_for_image:
        output_filename = os.path.splitext(file_name)[0] + ".pkl"
        output_filepath = os.path.join(output_directory, output_filename)
        with open(output_filepath, 'wb') as f_out:
            pickle.dump(results_for_image, f_out)

print("\nProcessing complete.")
print(f"Extracted features saved to: {output_directory}")