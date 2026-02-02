import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class EmojiDataset(Dataset):
    """Emoji dataset with text embeddings for conditional generation."""

    def __init__(self, image_size, data_dir=None):
        if data_dir is None:
            # 默认路径：从当前文件往上找 datasets 目录
            data_dir = os.path.join(os.path.dirname(__file__), "../../datasets")

        data_path = os.path.join(data_dir, "emoji_data.npz")
        text_emb_path = os.path.join(data_dir, "text_embeddings.pt")

        # 加载数据
        loaded = np.load(data_path, allow_pickle=True)
        self.data = [loaded[key].item() for key in loaded]

        # 加载预计算的 text embeddings
        text_emb_data = torch.load(text_emb_path, weights_only=False)
        self.idx_mapping = text_emb_data["idx_mapping"]
        self.text_embs = text_emb_data["embs"].float()

        # 图像预处理
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()  # [0, 255] -> [0, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        imgs = item["images"]
        texts = item["texts"]

        # 随机选择一张图片
        img_idx = np.random.randint(len(imgs))
        img = imgs[img_idx]
        img = Image.fromarray(img)
        img = self.transform(img)

        # 随机选择一个文本描述
        text_idx = np.random.randint(len(texts))
        text = texts[text_idx]

        # 获取预计算的 text embedding
        emb_idx = self.idx_mapping[text]
        text_emb = self.text_embs[emb_idx]

        model_kwargs = {
            "text_emb": text_emb,
            "text": text
        }
        return img, model_kwargs


def get_dataloader(image_size, batch_size, num_workers=4, data_dir=None):
    """创建 DataLoader 的便捷函数"""
    dataset = EmojiDataset(image_size, data_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader
