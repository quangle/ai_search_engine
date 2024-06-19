from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset
import json
import os
import faiss
import pandas as pd

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

def get_embedding(item):
    # Hàm thực hiện map
    # Nhiệm vụ:
    # tokenizer mẫu hiện tại
    tokens = tokenizer(
        item,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    # lấy vector embedidng của mẫu
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    counted = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counted
    print(f'embedding shape: {mean_pooled.shape}')
    return mean_pooled.detach().numpy()

class ReRanker():
    def __init__(self):
        f = open('docs.json')
        data = json.load(f)
        self.docs = data['docs']
        pass

    def rank(self, query, docs):
        # get_embedding for query
        query_embedding = get_embedding([query])
        print(f'query_embedding: {query_embedding}')

        # Load các docs từ TF-IDF và chuyển thành Hugging Dataset
        print(f'docs: {docs}')
        documents = [ {'id': int(doc[1]), 'text': self.docs[int(doc[1])] } for doc in docs]
        dataset_docs = Dataset.from_list(documents)
        # apply get_embedding to every dataset rows
        dataset_docs = dataset_docs.map(
            lambda x: {"embeddings": get_embedding(x['text'])[0]}
        )
        # add faiss index to embeddings column
        dataset_docs.add_faiss_index(column='embeddings')
        print(f'documents: {dataset_docs}')
        
        # Thực hiện tìm vector tương đồng với query
        scores, retrieved_examples = dataset_docs.get_nearest_examples('embeddings', query_embedding, k=10)
        print(f'scores: {scores}')
        print(f'retrieved_examples: {retrieved_examples}')

        samples_df = pd.DataFrame.from_dict(retrieved_examples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=False, inplace=True)
        result = []
        for _, row in samples_df.iterrows():
            result.append({
                'id': row.id,
                'score': row.scores,
                'text': row.text,
            })
        # Trả về kết quả
        return result
