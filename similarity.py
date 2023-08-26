from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from pprint import pprint

from sentence_transformers import SentenceTransformer


def get_data(input_file='input.txt'):
    with open(input_file, 'r') as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
    return sentences


def get_embeddings_hf(model_name, sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    inputs = tokenizer(sentences, return_tensors='pt', padding=True)

    result = model(**inputs)

    embeddings = result[0][:, 0, :]
    return embeddings


def get_embeddings_st(model_name, sentences):
    model = SentenceTransformer(model_name)

    embeddings = model.encode(sentences)
    return torch.from_numpy(embeddings)



def get_similarities(embeddings):
    similarities = []

    with torch.no_grad():
        for i, sentence1 in enumerate(embeddings):
            for j, sentence2 in enumerate(embeddings):
                if i < j:
                    similarity = F.cosine_similarity(sentence1.unsqueeze(0),
                            sentence2.unsqueeze(0)).item()

                    similarities.append((similarity, (sentences[i], sentences[j])))

    similarities = sorted(similarities, reverse=True)
    return similarities


if __name__ == '__main__':
    sentences = get_data()

    #model_name = 'distilroberta-base'
    #embeddings = get_embeddings_hf(model_name, sentences)

    model_name = 'all-distilroberta-v1'
    embeddings = get_embeddings_st(model_name, sentences)

    similarities = get_similarities(embeddings)

    print()
    pprint(similarities[:5])
