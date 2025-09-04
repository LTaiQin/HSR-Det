import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import json
from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 配置区域 (已优化)
# ==============================================================================
CONFIG = {
    # --- 文件路径 ---
    "coco_annotations_path": "/22liushoulong/datasets/ZSFooD2/annotations/instances_val2017.json",
    "descriptions_json_path": "/22liushoulong/projects3/ovfoodD_simple/preprocessing/descriptions_sentences_by_eval_new_cleaned.json",
    "base_embeddings_path": "/22liushoulong/projects3/ovfoodD_simple/datasets/zeroshot_weights/coco_clip_a+photo+cname.npy",
    "image_features_dir": "/22liushoulong/projects3/OVFD/zsfood_opt_des_emb/image_embedding_cache",
    "diff_embeddings_path": "/22liushoulong/projects3/ovfoodD_simple/ovd/modeling/CAR-Net/diff_embeddings.pt",

    # --- 模型参数 ---
    "clip_model_name": "ViT-B/32",
    "feature_dim": 512,
    "hidden_dim": 128,  # ## [优化] ## 简化模型，降低隐藏层维度以减少过拟合

    # --- 训练参数 (已优化) ---
    "epochs": 100,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,  # ## [优化] ## 增大权重衰减，加强正则化
    "train_val_split": 0.9,
    "noise_level": 0.01,  # ## [优化] ## 默认启用特征噪声，提升泛化能力
    "early_stopping_patience": 10,  # 早停耐心轮数

    # --- 输出 ---
    "save_path": "./car_net_best_optimized.pth"
}


# ==============================================================================
# 2. [优化后] CAR-Net 模块定义
# ==============================================================================
class CARNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        input_dim = feature_dim * 4
        # ## [优化] ## 简化为单隐藏层网络，降低过拟合风险
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, F_img, E_base_pred, E_base_confuser, E_diff):
        x = torch.cat((F_img, E_base_pred, E_base_confuser, E_diff), dim=1)
        return self.net(x)


# ==============================================================================
# 3. 数据加载与预处理函数
# ==============================================================================
def load_coco_classes(coco_json_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    print(f"成功从 {os.path.basename(coco_json_path)} 加载 {len(id_to_name)} 个类别。")
    return id_to_name, name_to_id


def load_base_embeddings(npy_path, id_to_name_map, device):
    npy_embeds = np.load(npy_path)
    base_embeddings = {}
    for class_id in id_to_name_map.keys():
        npy_index = class_id - 1
        if 0 <= npy_index < npy_embeds.shape[0]:
            base_embeddings[class_id] = torch.from_numpy(npy_embeds[npy_index]).to(device, dtype=torch.float32)
    print(f"成功加载 {len(base_embeddings)} 个基础类别嵌入。")
    return base_embeddings


def load_image_features(cache_dir, name_to_id_map, device):
    image_features = defaultdict(list)
    print(f"正在从 {cache_dir} 加载图像特征...")
    class_names_in_dir = [f.replace('.pt', '') for f in os.listdir(cache_dir) if f.endswith('.pt')]
    for class_name in tqdm(class_names_in_dir, desc="Scanning image folders"):
        if class_name in name_to_id_map:
            class_id = name_to_id_map[class_name]
            pt_path = os.path.join(cache_dir, f"{class_name}.pt")
            try:
                features_tensor = torch.load(pt_path, map_location=device)
                image_features[class_id].extend(list(features_tensor))
            except Exception as e:
                print(f"加载文件 {pt_path} 时出错: {e}")
    print(f"成功加载 {len(image_features)} 个类别的图像特征。")
    return dict(image_features)


def preprocess_and_load_diffs(descriptions_json_path, diff_embeddings_path, name_to_id_map):
    with open(descriptions_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 使用 defaultdict 来简化数据填充
    hard_pairs_database = defaultdict(dict)
    for class_id_str, descriptions in raw_data.items():
        for desc_string, confuser_info in descriptions.items():
            if 'confusers' in confuser_info and confuser_info['confusers']:
                hard_pairs_database[int(class_id_str)][desc_string] = confuser_info

    diff_embeddings = torch.load(diff_embeddings_path)
    print("成功加载困难对描述数据库和细粒度描述嵌入。")
    return hard_pairs_database, diff_embeddings


# ==============================================================================
# 4. 自定义数据集类
# ==============================================================================
class CARNetDataset(Dataset):
    def __init__(self, image_features, hard_pairs_database, all_base_embeddings, all_diff_embeddings, name_to_id_map,
                 noise_level=0.01):
        super().__init__()
        self.samples = []
        self.noise_level = noise_level
        self.is_train = True

        print("正在构建训练样本 (支持1-vs-N情境)...")
        for class_id_A, descriptions in tqdm(hard_pairs_database.items(), desc="Generating training samples"):
            for desc_string, confuser_info in descriptions.items():
                class_name_B = list(confuser_info['confusers'].keys())[0]
                if class_name_B not in name_to_id_map: continue
                class_id_B = name_to_id_map[class_name_B]

                # 正样本: A的图片，应该匹配 D(A vs B)
                if class_id_A in image_features:
                    for F_img in image_features[class_id_A]:
                        self.samples.append((F_img, class_id_A, class_id_B, desc_string, 1.0))

                # 负样本: B的图片，不应该匹配 D(A vs B)
                if class_id_B in image_features:
                    for F_img in image_features[class_id_B]:
                        self.samples.append((F_img, class_id_B, class_id_A, desc_string, -1.0))

        self.all_base_embeddings = all_base_embeddings
        self.all_diff_embeddings = all_diff_embeddings
        print(f"CAR-Net 数据集创建成功，共包含 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        F_img, pred_id, confuser_id, diff_str, target = self.samples[idx]

        if self.is_train and self.noise_level > 0:
            noise = torch.randn_like(F_img) * self.noise_level
            F_img = F_img + noise

        E_base_pred = self.all_base_embeddings[pred_id]
        E_base_confuser = self.all_base_embeddings[confuser_id]
        E_diff = self.all_diff_embeddings[diff_str]
        return F_img, E_base_pred, E_base_confuser, E_diff, torch.tensor(target, dtype=torch.float32)


# ==============================================================================
# 5. 训练与验证函数
# ==============================================================================
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for F_img, E_base_pred, E_base_confuser, E_diff, targets in pbar:
        F_img, E_base_pred, E_base_confuser, E_diff, targets = \
            F_img.to(device), E_base_pred.to(device), E_base_confuser.to(device), E_diff.to(device), targets.to(device)
        optimizer.zero_grad()
        scores = model(F_img, E_base_pred, E_base_confuser, E_diff)
        loss = loss_fn(scores.squeeze(-1), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct_predictions, total_samples = 0, 0, 0
    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for F_img, E_base_pred, E_base_confuser, E_diff, targets in pbar:
            F_img, E_base_pred, E_base_confuser, E_diff, targets = \
                F_img.to(device), E_base_pred.to(device), E_base_confuser.to(device), E_diff.to(device), targets.to(
                    device)
            scores = model(F_img, E_base_pred, E_base_confuser, E_diff)
            loss = loss_fn(scores.squeeze(-1), targets)
            total_loss += loss.item()
            predicted_signs = torch.sign(scores.squeeze(-1))
            correct_predictions += (predicted_signs == targets).sum().item()
            total_samples += targets.size(0)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return total_loss / len(dataloader), accuracy


# ==============================================================================
# 6. 主执行函数
# ==============================================================================
if __name__ == "__main__":
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"使用计算设备: {gpu_device}, 使用数据存储设备: {cpu_device}")

    # --- 1. 加载所有数据 ---
    print("\n步骤 1: 加载所有数据...")
    id_to_name_map, name_to_id_map = load_coco_classes(CONFIG["coco_annotations_path"])
    base_embeddings = load_base_embeddings(CONFIG["base_embeddings_path"], id_to_name_map, cpu_device)
    image_features = load_image_features(CONFIG["image_features_dir"], name_to_id_map, cpu_device)

    if not os.path.exists(CONFIG["diff_embeddings_path"]):
        print(f"\n错误: 未找到细粒度描述嵌入文件: {CONFIG['diff_embeddings_path']}")
        print("请先运行脚本来生成此文件。")
        exit()

    hard_pairs_database, diff_embeddings = preprocess_and_load_diffs(
        CONFIG["descriptions_json_path"], CONFIG["diff_embeddings_path"], name_to_id_map
    )

    # --- 2. 创建数据集 ---
    print("\n步骤 2: 创建 PyTorch 数据集与加载器...")
    full_dataset = CARNetDataset(
        image_features, hard_pairs_database, base_embeddings, diff_embeddings, name_to_id_map,
        noise_level=CONFIG["noise_level"]
    )
    if len(full_dataset) == 0:
        print("\n错误：最终生成的数据集为空！请检查您的数据文件和路径。")
        exit()

    train_size = int(CONFIG["train_val_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

    train_subset.dataset.is_train = True
    val_subset.dataset.is_train = False

    train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. 初始化模型和优化器 ---
    print("\n步骤 3: 初始化 CAR-Net, 损失函数和优化器...")
    model = CARNet(feature_dim=CONFIG["feature_dim"], hidden_dim=CONFIG["hidden_dim"]).to(gpu_device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=True)

    # --- 4. 开始训练 ---
    print("\n" + "=" * 20 + " 开始训练优化的 CAR-Net " + "=" * 20)
    best_val_accuracy = -1.0
    epochs_no_improve = 0

    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, gpu_device)
        val_loss, val_accuracy = validate_epoch(model, val_loader, loss_fn, gpu_device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1} 结果 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            print(f"  -> 新的最佳验证集准确率: {best_val_accuracy:.4f}。正在保存模型至 {CONFIG['save_path']}...")
            torch.save(model.state_dict(), CONFIG['save_path'])
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG["early_stopping_patience"]:
            print(f"\n验证集准确率连续 {epochs_no_improve} 个 epoch 未提升，触发早停。")
            break

    print("\n" + "=" * 20 + " 训练完成 " + "=" * 20)
    print(f"最佳验证集准确率: {best_val_accuracy:.4f}")
    print(f"最优 CAR-Net 模型已保存在: {CONFIG['save_path']}")