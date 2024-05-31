import sys
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize

# Load embeddings
file_path = 'Application//glove.6B.100d.txt'

def load_glove_embeddings(file_path):
    embeddings_index = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        print("Found %s word vectors." % len(embeddings_index))
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please ensure the file is in the correct directory.")
    except Exception as e:
        print("An error occurred:", e)
    return embeddings_index

# Compute sentence embeddings
def sentence_embeddings(text, embeddings_index):
    sentence_tokens = sent_tokenize(text)
    sentence_embed = []
    
    for sent in sentence_tokens:
        word_tokens = word_tokenize(sent)
        embeddings = []
        
        for word in word_tokens:
            if word in embeddings_index:
                embeddings.append(embeddings_index[word])

        if embeddings:
            sentence_embed.append(np.mean(embeddings, axis=0))
        else:
            sentence_embed.append(np.zeros_like(next(iter(embeddings_index.values()))))
    
    return sentence_embed, sentence_tokens

# Compute cosine similarity matrix
def cosine_similarity_matrix(sentence_embeddings):
    similarity_matrix = np.zeros((len(sentence_embeddings), len(sentence_embeddings)))
    for i in range(len(sentence_embeddings)):
        for j in range(len(sentence_embeddings)):
            similarity_matrix[i, j] = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j]))
    return similarity_matrix

def lexrank_summary(text, embeddings_index, threshold=0.6, num_sentences_in_summary=5):
    sentence_embed, sentence_tokens = sentence_embeddings(text, embeddings_index)
    similarity_matrix = cosine_similarity_matrix(sentence_embed)
    
    G = nx.Graph()
    num_sentences = len(sentence_embed)
    G.add_nodes_from(range(num_sentences))

    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > threshold:
                G.add_edge(i, j, weight=similarity_score)

    scores = nx.pagerank(G)
    top_sentence_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences_in_summary]
    summary = [sentence_tokens[i] for i in top_sentence_indices]
    summary_text = ' '.join(summary)
    
    return summary_text
