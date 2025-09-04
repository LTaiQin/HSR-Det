import os
import cv2
import torchvision
from typing import List, Dict, Optional
from detectron2.structures import Instances, Boxes

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
    print(f"图片将保存到: {output_dir}")

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

        print(f"正在处理图片: {img_path}，原始框: {len(boxes)}个 -> NMS后: {len(final_boxes)}个...")

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

    print("所有图片处理完毕。")