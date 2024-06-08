from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset

import os

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

def get_embedding(item):
    # Hàm thực hiện map
    # Nhiệm vụ:
    # tokenizer mẫu hiện tại

    # lấy vector embedidng của mẫu
    pass


class ReRanker():
    def __init__(self):
        pass

    def rank(self, query, docs):

        # Load các docs từ TF-IDF và chuyển thành Hugging Dataset
        # Thực hiện tìm vector tương đồng với query
        # Trả về kết quả
        return []