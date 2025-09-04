# file: create_diff_embeddings.py

import torch
import clip
import json
from tqdm import tqdm
import os

# --- 配置区域 ---
CONFIG = {
    "descriptions_json_path": "/22liushoulong/projects3/ovfoodD_simple/preprocessing/descriptions_sentences_by_eval_new_cleaned.json",
    "output_path": "./diff_embeddings.pt",  # 这是您要生成的目标文件
    "clip_model_name": "ViT-B/32",
    "batch_size": 256  # 批量编码以节省显存
}


def generate_and_save_diff_embeddings(cfg):
    """
    读取描述JSON，用CLIP编码，并保存为.pt文件。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载CLIP模型
    clip_model, _ = clip.load(cfg["clip_model_name"], device=device)

    # 2. 读取并收集所有唯一的描述字符串
    print(f"正在从 {cfg['descriptions_json_path']} 读取描述...")
    with open(cfg["descriptions_json_path"], 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    all_diff_strings = set()
    for _, descriptions in raw_data.items():
        for desc_string in descriptions.keys():
            all_diff_strings.add(desc_string)

    unique_strings = list(all_diff_strings)
    print(f"找到 {len(unique_strings)} 条唯一的细粒度描述，准备编码...")

    # 3. 批量编码
    diff_embeddings = {}
    total_batches = (len(unique_strings) + cfg["batch_size"] - 1) // cfg["batch_size"]

    with torch.no_grad():
        for i in tqdm(range(total_batches), desc="Encoding Descriptions"):
            batch_strings = unique_strings[i * cfg["batch_size"]: (i + 1) * cfg["batch_size"]]
            if not batch_strings:
                continue

            text_tokens = clip.tokenize(batch_strings).to(device)
            text_features = clip_model.encode_text(text_tokens)

            for j, text_str in enumerate(batch_strings):
                # 将嵌入保存在CPU上，以方便后续加载
                diff_embeddings[text_str] = text_features[j].cpu().to(dtype=torch.float32)

    # 4. 保存结果
    torch.save(diff_embeddings, cfg["output_path"])
    print(f"\n成功生成细粒度描述嵌入文件！")
    print(f"文件已保存在: {cfg['output_path']}")


if __name__ == "__main__":
    generate_and_save_diff_embeddings(CONFIG)