import torch
import torch.nn.functional as F
import clip
import json
from itertools import combinations


class DynamicRefinementModule:
    def __init__(self, difficult_pairs_path, descriptions_path, annotations_path,
                 norm_temperature=50.0,  # <--- 新增参数
                 clip_model_name="ViT-B/32", device="cuda", top_k=3):
        """
        初始化动态精炼模块

        :param norm_temperature: 与主模型一致的温度系数 (logit_scale)
        ... 其他参数不变 ...
        """
        print("--- Initializing Dynamic Refinement Module ---")
        self.device = device
        self.top_k = top_k
        self.norm_temperature = norm_temperature  # <--- 保存温度系数

        # ... 其他初始化代码不变 ...
        # 1. 加载CLIP模型
        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()

        # 2. 加载类别ID和名称映射
        print("Loading class maps...")
        with open(annotations_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        self.id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}

        # 3. 加载并处理困难样本对
        print("Loading difficult pairs...")
        with open(difficult_pairs_path, 'r', encoding='utf-8') as f:
            difficult_pairs_list = json.load(f)
        self.difficult_pairs_set = set()
        for pair in difficult_pairs_list:
            self.difficult_pairs_set.add(frozenset([pair['class_name_1'], pair['class_name_2']]))

        # 4. 加载描述数据库
        print("Loading descriptions database...")
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            self.descriptions_db = {int(k): v for k, v in json.load(f).items()}

        print("--- Module Initialized Successfully ---")

    # ... _get_contrastive_sentence 方法不变 ...
    def _get_contrastive_sentence(self, target_class_id, confuser_class_name):
        """从数据库中查找最具区分度的描述句"""
        if target_class_id not in self.descriptions_db:
            return None
        best_sentence = None
        max_score = -float('inf')
        for sentence, data in self.descriptions_db[target_class_id].items():
            if confuser_class_name in data.get("confusers", {}):
                score = data["confusers"][confuser_class_name]
                if score > max_score:
                    max_score = score
                    best_sentence = sentence
        return best_sentence

    @torch.no_grad()
    def refine(self, image_embeddings, initial_scores):
        """
        对初步的分类结果进行精炼

        :param image_embeddings: 图像区域嵌入, Tensor(N, 512)。注意：这里应该是【未经过】温度缩放的归一化嵌入。
        :param initial_scores: 初始的分类分数, Tensor(N, M)
        :return: 精炼后的分类分数, Tensor(N, M)
        """

        refined_scores = initial_scores.clone()
        num_boxes = image_embeddings.shape[0]

        _, topk_indices = torch.topk(initial_scores, self.top_k, dim=1)

        for i in range(num_boxes):
            candidate_ids = [idx.item() + 1 for idx in topk_indices[i]]
            candidate_names = [self.id_to_name.get(cid) for cid in candidate_ids]

            found_difficult_pairs = []
            for name1, name2 in combinations(candidate_names, 2):
                if name1 and name2 and frozenset([name1, name2]) in self.difficult_pairs_set:
                    found_difficult_pairs.append((name1, name2))

            if not found_difficult_pairs:
                continue

            # ======================= [核心修改] =======================
            # 确保用于计算新分数的图像嵌入与原始计算方式一致
            # 你的原始计算是 x = self.norm_temperature * F.normalize(x, p=2, dim=1)
            # 所以我们这里也需要一个归一化的图像嵌入
            img_emb_normalized = F.normalize(image_embeddings[i].unsqueeze(0), p=2, dim=1)
            # =========================================================

            for name_A, name_B in found_difficult_pairs:
                id_A, id_B = self.name_to_id[name_A], self.name_to_id[name_B]
                idx_A, idx_B = id_A - 1, id_B - 1

                sent_A_vs_B = self._get_contrastive_sentence(id_A, name_B)
                prompt_A = f"a photo of {sent_A_vs_B}" if sent_A_vs_B else f"a photo of {name_A}"
                token_A = clip.tokenize(prompt_A, truncate=True).to(self.device)
                emb_A_enhanced = self.clip_model.encode_text(token_A)

                sent_B_vs_A = self._get_contrastive_sentence(id_B, name_A)
                prompt_B = f"a photo of {sent_B_vs_A}" if sent_B_vs_A else f"a photo of {name_B}"
                token_B = clip.tokenize(prompt_B, truncate=True).to(self.device)
                emb_B_enhanced = self.clip_model.encode_text(token_B)

                # 对增强文本嵌入也进行归一化，与标准CLIP流程保持一致
                emb_A_enhanced = F.normalize(emb_A_enhanced, p=2, dim=1)
                emb_B_enhanced = F.normalize(emb_B_enhanced, p=2, dim=1)

                # ======================= [核心修改] =======================
                # 计算新分数时，应用完全相同的计算逻辑
                new_score_A = torch.mm(img_emb_normalized, emb_A_enhanced.T) * self.norm_temperature
                new_score_B = torch.mm(img_emb_normalized, emb_B_enhanced.T) * self.norm_temperature
                # =========================================================

                refined_scores[i, idx_A] = new_score_A.squeeze()
                refined_scores[i, idx_B] = new_score_B.squeeze()

        return refined_scores

# ==================== 如何在你的代码中调用 ====================

# 在你的主程序或评估脚本的初始化阶段
# refinement_module = DynamicRefinementModule(
#     difficult_pairs_path="difficult_pairs_final.json",
#     descriptions_path="descriptions_sentences_global_new.json",
#     annotations_path="/22liushoulong/datasets/ZSFooD2/annotations/instances_val2017.json",
#     norm_temperature=50.0, # <-- 传入你的温度系数
#     device="cuda"
# )

# 在你的模型推理循环中
# ... 得到【原始的】图像嵌入 raw_image_embeddings (Tensor N*512) ...
#
# # 步骤1: 计算初始分数 (和你现在的代码一样)
# image_embeddings_processed = self.norm_temperature * F.normalize(raw_image_embeddings, p=2, dim=1)
# initial_scores = torch.mm(image_embeddings_processed, self.zs_weight)
#
# # 步骤2: 调用精炼模块
# # 注意：传递给refine函数的应该是【未经过温度缩放】的归一化图像嵌入，或者干脆是原始嵌入
# # 为了清晰，我们这里传递原始嵌入，让模块内部自己处理归一化
# final_scores = refinement_module.refine(raw_image_embeddings, initial_scores)
#
# # 步骤3: 得到最终预测
# final_predictions = torch.argmax(final_scores, dim=1)