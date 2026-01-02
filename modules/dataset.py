"""
数据集模块 - BigFiveDataset类
"""

import os
import json
import re
import hashlib
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import BertTokenizer
from typing import List, Dict, Tuple

from .config import get_model_path

try:
    from data.BigFive_definition import definition
except ImportError:
    print("警告：无法导入 BigFive_definition.py，将使用默认定义")
    definition = {
        '开放性': '开放性是指对新经验、新想法的接受程度',
        '尽责性': '尽责性是指做事认真负责、有条理的程度',
        '外向性': '外向性是指与他人互动、寻求刺激的倾向',
        '宜人性': '宜人性是指与他人和谐相处、关心他人的程度',
        '神经质性': '神经质性是指情绪不稳定、容易焦虑的倾向'
    }


class BigFiveDataset(Dataset):
    """改进的大五人格数据集 - 使用PyG Data格式"""

    def __init__(self, data_path: str, use_bert: bool = True, max_length: int = 64):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径
            use_bert: 是否使用BERT嵌入
            max_length: 文本最大长度
        """
        self.data_path = data_path
        self.use_bert = use_bert
        self.max_length = max_length
        self.data = self._load_data()

        self.tokens_dir = os.path.join(os.path.dirname(self.data_path), 'tokens')
        os.makedirs(self.tokens_dir, exist_ok=True)

        if self.use_bert:
            try:
                model_path = get_model_path()
                print(f"从本地路径加载tokenizer: {model_path}")
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"Tokenizer加载失败: {e}")
                raise RuntimeError(f"无法加载BERT tokenizer: {e}")

        print("正在计算人格节点分词...")
        self.personality_input_ids, self.personality_attention_mask = self._compute_personality_tokens()

    def _load_data(self) -> List[Dict]:
        """
        加载数据集

        Returns:
            data: 数据列表
        """
        print(f"正在加载数据: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载完成，共 {len(data)} 条数据")
        return data

    def _compute_personality_tokens(self):
        """
        计算人格节点的分词编码

        Returns:
            personality_input_ids: 人格节点的输入ID
            personality_attention_mask: 人格节点的注意力掩码
        """
        personalities = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
        input_ids_list = []
        attention_mask_list = []
        for personality in personalities:
            definition_text = definition.get(personality)
            tokens = self.tokenizer(
                definition_text,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            input_ids_list.append(tokens['input_ids'].squeeze(0))
            attention_mask_list.append(tokens['attention_mask'].squeeze(0))
        return torch.stack(input_ids_list), torch.stack(attention_mask_list)

    def _encode_text(self, text: str):
        """
        对文本进行编码，支持缓存

        Args:
            text: 要编码的文本

        Returns:
            input_ids: 输入ID
            attention_mask: 注意力掩码
        """
        key = text if text and text.strip() else ""
        text_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()
        cache_file = os.path.join(self.tokens_dir, f"{text_hash}.pt")

        if os.path.exists(cache_file):
            try:
                d = torch.load(cache_file)
                return d['input_ids'], d['attention_mask']
            except Exception:
                pass

        tokens = self.tokenizer(
            key,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        ids = tokens['input_ids'].squeeze(0)
        mask = tokens['attention_mask'].squeeze(0)

        try:
            torch.save({'input_ids': ids, 'attention_mask': mask}, cache_file)
        except Exception:
            pass

        return ids, mask

    def _parse_dialogue(self, dialogue_text: str, target_character: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
        """
        彻底重构对话解析逻辑，确保所有句子都被完整解析。

        1. 使用正则表达式安全地分割句子，保留"第X句"作为后续处理的锚点。
        2. 对每个句子片段，使用更精确的正则提取说话人和内容。

        Args:
            dialogue_text: 对话文本
            target_character: 目标角色名

        Returns:
            target_utterances: 目标角色的发言列表
            other_utterances: 其他角色的发言列表
            all_utterances: 所有发言的(说话人, 内容)列表
        """
        if not dialogue_text or not isinstance(dialogue_text, str):
            return [], [], []

        utterances = re.split(r'(?=第\d+句)', dialogue_text)
        utterances = [u.strip() for u in utterances if u and u.strip()]

        target_utterances = []
        other_utterances = []
        all_utterances = []

        for utterance in utterances:
            match = re.match(r'第\d+句(\S+?)(?:说)[：:](.+)', utterance, re.DOTALL)

            if match:
                speaker, content = match.groups()
                speaker = speaker.strip()
                content = content.strip()
                all_utterances.append((speaker, content))

                normalized_speaker = re.sub(r'[\s（）（）]', '', speaker)

                if normalized_speaker == target_character:
                    target_utterances.append(content)
                else:
                    other_utterances.append(content)

        return target_utterances, other_utterances, all_utterances

    def _create_pyg_data(self, item: Dict) -> Data:
        """
        创建PyG Data对象

        Args:
            item: 数据项

        Returns:
            data: PyG Data对象
        """
        dialogue_text = item.get('dialogue')
        target_character = item.get('target_character')

        target_utterances, other_utterances, all_utterances = self._parse_dialogue(dialogue_text, target_character)

        num_target = len(target_utterances)
        num_other = len(other_utterances)
        num_all = len(all_utterances)

        num_personality = 5
        total_nodes = num_all + num_personality

        personality_start = num_all
        adj = torch.zeros((total_nodes, total_nodes), dtype=torch.bool)

        for i in range(num_all - 1):
            adj[i, i + 1] = True
            adj[i + 1, i] = True

        last_idx = {}
        for idx, (speaker, _) in enumerate(all_utterances):
            if speaker in last_idx:
                prev = last_idx[speaker]
                adj[prev, idx] = True
                adj[idx, prev] = True
            last_idx[speaker] = idx

        for idx, (speaker, _) in enumerate(all_utterances):
            if speaker == target_character:
                for j in range(5):
                    p = personality_start + j
                    adj[idx, p] = True
                    adj[p, idx] = True

        for i in range(5):
            for j in range(5):
                if i != j:
                    a = personality_start + i
                    b = personality_start + j
                    adj[a, b] = True
                    adj[b, a] = True

        seq_ids_list, seq_mask_list = [], []
        for speaker, content in all_utterances:
            ids, mask = self._encode_text(content)
            seq_ids_list.append(ids)
            seq_mask_list.append(mask)
        if seq_ids_list:
            utter_input_ids = torch.stack(seq_ids_list)
            utter_attention_mask = torch.stack(seq_mask_list)
        else:
            print("警告：当前样本无发言，查找错误！！！")

        num_all = len(all_utterances)
        all_input_ids = torch.cat([utter_input_ids, self.personality_input_ids], dim=0)
        all_attention_mask = torch.cat([utter_attention_mask, self.personality_attention_mask], dim=0)
        personality_start = num_all
        total_nodes = num_all + 5

        personality_mask = torch.zeros(total_nodes, dtype=torch.bool)
        personality_mask[personality_start:] = True

        labels_vector = item.get('labels_vector', [])
        if len(labels_vector) != 15:
            print("警告：标签长度不为15，查找错误！！！")

        labels = torch.tensor(labels_vector, dtype=torch.float)

        utter_ids_raw = item.get('utter_ids', {})

        data = Data(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            personality_mask=personality_mask,
            labels=labels,
            target_character=item['target_character'],
            dialogue_id=item['dialogue_id'],
            num_target=num_target,
            num_other=num_other,
            num_nodes=total_nodes,
            dialogue=dialogue_text,
            adjacency_matrix=adj,
            utter_ids=utter_ids_raw
        )

        return data

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据项

        Args:
            idx: 索引

        Returns:
            data: PyG Data对象
        """
        try:
            return self._create_pyg_data(self.data[idx])
        except Exception as e:
            print(f"处理第 {idx} 条数据时出错: {e}")
