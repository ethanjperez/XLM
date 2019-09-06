import faiss
import json
import numpy as np
import os
import time
from spacy.language import Language
from tqdm import tqdm

DATA_DIR = os.getenv('HOME') + '/research/data'


def main():
    # Load word embeddings
    print('Loading word embeddings...')
    nlp = Language()
    vector_filename = 'crawl-300d-2M.vec'  # wiki-news-300d-1M.vec
    with open(f'../fastText/pretrained/{vector_filename}', 'rb') as f:
        header = f.readline()
        nr_row, nr_dim = header.split()
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in tqdm(f, total=2000000):
            line = line.rstrip().decode("utf8")
            pieces = line.rsplit(" ", int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype="f")
            nlp.vocab.set_vector(word, vector)

    # Load questions and convert to L2-normalized vectors
    print('Loading questions...')
    with open(f'{DATA_DIR}/UnsupervisedQAData/train.json') as f:
        data = json.load(f)

    qs = []
    raw_qs = []
    for article in tqdm(data['data']):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                raw_q = qa['question']
                raw_qs.append(raw_q)
                token_q = nlp(raw_q)
                qs.append(token_q.vector / token_q.vector_norm)
    qs = np.array(qs)

    # TODO: Write q_vectors to file for FAISS

    index = faiss.IndexFlatIP(300)
    index.add(qs)
    print(f'Total Qs indexed: {index.ntotal}')

    k = 4
    D, I = index.search(qs[:5], k)
    print('NN Indexes:')
    print(I)
    print('NN Distances:')
    print(D)
    for i in range(5):
        # assert I[i, 0] == i, f'Q{i} is Q{I[i, 0]}, not itself!'
        # assert D[i, 0] == 1., f'Q{i} NN distance {D[i, 0]}, not 0!'
        print(f'Original Q: {raw_qs[i]}')
        for nn_rank in range(k):
            print(f'NN{f} of Q: {raw_qs[I[i, nn_rank]]} (Dist: {D[i, nn_rank]})')

    start_time = time.time()
    D, I = index.search(qs, 2)     # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    print(I[-5:])                  # neighbors of the 5 last queries
    print(f'Took {round(time.time() - start_time, 2)}s')
    print('Done!')


if __name__ == "__main__":
    main()
