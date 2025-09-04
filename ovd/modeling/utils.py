import torch
import json
import numpy as np
from torch.nn import functional as F
import os
import cv2
import torchvision
from typing import List, Dict, Optional
from detectron2.structures import Instances, Boxes
from collections import defaultdict

def load_class_freq(
        path='datasets/lvis/lvis_v1_train_norare_cat_info.json', freq_weight=1.0):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes)  # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


def reset_cls_test(model, cls_path, num_classes):
    model.roi_heads.num_classes = num_classes
    if type(cls_path) == str:
        print('Resetting zs_weight', cls_path)
        zs_weight = torch.tensor(
            np.load(cls_path),
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C
    else:
        zs_weight = cls_path
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))],
        dim=1)  # D x (C + 1)
    if model.roi_heads.box_predictor.cls_score.norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)
    del model.roi_heads.box_predictor.cls_score.zs_weight
    model.roi_heads.box_predictor.cls_score.zs_weight = zs_weight


def load_base_embeddings(npy_path, id_to_name_map, device):
    """加载Numpy格式的基础类别嵌入到指定设备。 (已修正索引逻辑)"""
    npy_embeds = np.load(npy_path)
    base_embeddings = {}
    print(f"正在从 {npy_path} 加载基础类别嵌入...")

    for class_id in id_to_name_map.keys():
        # **核心修正**: 将从1开始的class_id映射到从0开始的numpy数组索引
        npy_index = class_id - 1

        if 0 <= npy_index < npy_embeds.shape[0]:
            base_embeddings[class_id] = torch.from_numpy(npy_embeds[npy_index]).to(device, dtype=torch.float32)
        else:
            print(f"警告: 类别 ID {class_id} 在 .npy 文件中没有对应的嵌入 (索引 {npy_index} 超出范围)。")

    print(f"成功加载 {len(base_embeddings)} 个基础类别嵌入。")
    return base_embeddings

def load_coco_classes(coco_json_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    print(f"成功从 {os.path.basename(coco_json_path)} 加载 {len(id_to_name)} 个类别。")
    return id_to_name, name_to_id

def draw_and_save_boxes_with_nms(
        instances_list: List[Instances],
        batched_inputs: List[Dict],
        output_dir: str = "/22liushoulong/projects3/simple_model/output/images",
        score_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        class_names: Optional[List[str]] = None
):
    """
    对检测框进行NMS过滤后，将其绘制到对应的图片上，并保存到指定目录。

    Args:
        instances_list (List[Instances]): 模型预测结果的列表，列表长度为批次大小。
            每个元素是一个Instances对象，应包含 'pred_boxes', 'scores', 'pred_classes' 字段。
        batched_inputs (List[Dict]): 输入数据的列表，列表长度为批次大小。
            每个元素是一个字典，包含 'file_name' 键。
        output_dir (str): 保存绘制了边界框的图片的目录路径。
        score_thresh (float): 分数阈值，低于此分数的检测框将被过滤。
        iou_thresh (float): NMS的IoU阈值。
        class_names (Optional[List[str]]): 类别名称列表，用于在图上显示名称而非索引。
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (inputs, instances) in enumerate(zip(batched_inputs, instances_list)):
        img_path = inputs.get('file_name')
        if not img_path or not os.path.exists(img_path):
            print(f"警告: 第 {i} 个输入的图片路径无效或文件不存在: {img_path}，已跳过。")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图片: {img_path}，已跳过。")
            continue

        # 检查instances是否为空
        if len(instances) == 0:
            print(f"正在处理图片: {img_path}，但无任何检测结果。")
            continue

        # 提取必要信息
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        classes = instances.pred_classes

        # 1. 按分数阈值过滤
        keep_mask = scores > score_thresh
        boxes_filtered = boxes[keep_mask]
        scores_filtered = scores[keep_mask]
        classes_filtered = classes[keep_mask]

        if len(boxes_filtered) == 0:
            print(f"正在处理图片: {img_path}，无检测框通过分数阈值 {score_thresh}。")
            continue

        # 2. 执行NMS
        # torchvision.ops.nms需要boxes和scores
        keep_indices = torchvision.ops.nms(boxes_filtered, scores_filtered, iou_thresh)

        final_boxes = boxes_filtered[keep_indices].cpu()
        final_scores = scores_filtered[keep_indices].cpu()
        final_classes = classes_filtered[keep_indices].cpu()


        # 3. 遍历最终的边界框并在图上绘制
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = map(int, box)

            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # 准备要显示的文本
            class_id = int(cls_id)
            score_val = float(score)
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {score_val:.2f}"
            else:
                label = f"Class {class_id}: {score_val:.2f}"

            # 绘制文本背景
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
            # 绘制文本
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 构建并保存图片
        base_filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"pred_{base_filename}")
        cv2.imwrite(save_path, image)


def load_image_features(cache_dir, name_to_id_map):
    # This function is used for offline training.
    # For inference, features are passed in real-time.
    pass  # In this inference-focused refactor, this function is not needed by CARNetManager.


def preprocess_and_load_diffs(descriptions_json_path, diff_embeddings_path, name_to_id_map):
    # For inference, we assume embeddings are pre-computed and saved.
    with open(descriptions_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    hard_pairs_lookup = defaultdict(lambda: {"desc_A_vs_B": None, "desc_B_vs_A": None})
    for class_id_1_str, descriptions in raw_data.items():
        for desc_string, confuser_info in descriptions.items():
            if 'confusers' in confuser_info and confuser_info['confusers']:
                class_name_2 = list(confuser_info['confusers'].keys())[0]
                if class_name_2 not in name_to_id_map: continue
                class_id_2 = name_to_id_map[class_name_2]
                id1, id2 = sorted([int(class_id_1_str), int(class_id_2)])
                key = frozenset({id1, id2})
                if int(class_id_1_str) == id1:
                    hard_pairs_lookup[key]["desc_A_vs_B"] = desc_string
                else:
                    hard_pairs_lookup[key]["desc_B_vs_A"] = desc_string

    hard_pairs_database = dict(hard_pairs_lookup)
    diff_embeddings = torch.load(diff_embeddings_path)

    print("成功加载困难对数据库和细粒度描述嵌入。")
    return hard_pairs_database, diff_embeddings


def load_representative_features(npy_path, id_to_name_map, device):
    """加载Numpy格式的代表性类别图片特征。"""
    npy_embeds = np.load(npy_path)
    rep_features = {}
    print(f"正在从 {npy_path} 加载代表性图片特征...")

    for class_id in id_to_name_map.keys():
        # 核心: 将从1开始的class_id映射到从0开始的numpy数组索引
        npy_index = class_id - 1

        if 0 <= npy_index < npy_embeds.shape[0]:
            rep_features[class_id] = torch.from_numpy(npy_embeds[npy_index]).to(device, dtype=torch.float32)
        else:
            print(f"警告: 类别 ID {class_id} 在代表性特征文件 .npy 中没有对应的嵌入。")

    print(f"成功加载 {len(rep_features)} 个类别的代表性图片特征。")
    return rep_features