import codecs
import torch
from treelstm import SimilarityTreeLSTM, utils
import os
import numpy as np
from tqdm import tqdm

from treelstm.chunk_embedding import ChunkEmbedding
from gensim.models.keyedvectors import Word2VecKeyedVectors


def __read_text(fpath):
    with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
        lines = fp.readlines()
        lines = [line.split("\n")[0].strip().lower() for line in lines]
    return lines


def __read_chunks(fpath):
    with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
        chunks = fp.readlines()
        chunks = [line.split("\n")[0].strip().lower() for line in chunks]
        chunks = [l[1:-1].split("] [") for l in chunks]
        chunks = [[e.strip() for e in l] for l in chunks]
        chunks = [[e.split() for ix, e in enumerate(l)] for l in chunks]
        return chunks


def _write_vectors(vectors, output_file):
    emb_vectors = Word2VecKeyedVectors(vector_size=150)
    emb_vectors.vocab = vectors
    emb_vectors.vectors = np.array(vectors.values())
    emb_vectors.save(output_file)


def __generate_embeddings(plain_file, chunk_file, chunk_emb_model, file_identifier, error_writer):
    lines = __read_text(plain_file)
    chunks_list = __read_chunks(chunk_file)
    vectors = {}

    for idx in tqdm(xrange(len(lines))):
        sentence = lines[idx]
        chunks = chunks_list[idx]

        try:
            const_tree, idx_span_mapping = chunk_emb_model.constituent_parse(sentence)
        except Exception as e:
            error_writer.write("CP_ERROR_" + file_identifier + ":" + sentence + "\n")
            # print "constituent_parse failed : ", sentence
            continue

        start_chunk_idx = 0
        for chunk in chunks:
            end_chunk_idx = start_chunk_idx + len(chunk)
            emb = chunk_emb_model.get_embedding(sentence, chunk, const_tree, idx_span_mapping)
            if torch.all(torch.eq(emb, torch.zeros(1, 1))).numpy().item() == 1:
                # print "Skipping chunk : ", chunk, " : for sentence : ", sentence
                error_writer.write("INVALID_CHUNK_" + file_identifier + ":" + sentence + "__" + ' '.join(chunk) + "\n")
            vectors[str(idx+1) + "_" + file_identifier + '_' + ','.join(map(str, range(start_chunk_idx+1, end_chunk_idx+1)))] = emb.numpy().ravel()
            start_chunk_idx = end_chunk_idx

    return vectors


if __name__ == '__main__':

    base_path = "/mnt/manish/treelstm_with_changes/"
    pickle_file = base_path + "checkpoints/best_model_treelstm_changes.pt"
    glove_vocab, glove_emb_matrix = utils.load_keyed_vectors(base_path + "data/glove/glove.840B.300d")

    model = SimilarityTreeLSTM(weight_matrix=glove_emb_matrix, in_dim=300, mem_dim=150, hidden_dim=50, num_classes=5, sparsity=False,
                               freeze=True, use_parse_tree="constituency")

    lib_dir = base_path + "lib"
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    temp_dir = base_path + "temp"

    chunk_embedding = ChunkEmbedding(model, pickle_file, glove_vocab, classpath, temp_dir, 123)

    #################################
    # plain_file = "/Users/manish.bansal/repos/search/treelstm.pytorch/dummy_sent1.txt"
    # chunk_file = "/Users/manish.bansal/repos/search/treelstm.pytorch/dummy_chunk.txt"
    # output_file = "/Users/manish.bansal/repos/search/treelstm.pytorch/dummy_output.txt"

    file_prefixes = ["STSint.input.headlines.", "STSint.testinput.headlines."]
    files = [("sent1.txt", "sent1.chunk.txt", "l"), ("sent2.txt", "sent2.chunk.txt", "r")]

    error_writer = codecs.open(base_path + "output/error.log", mode='w', encoding='utf-8', errors='ignore')

    for f_prefix in file_prefixes:
        vectors_combined = {}
        output_file = base_path + "output/" + f_prefix + "_emb.bin"
        for f1, f2, idx in files:
            plain_file = base_path + "input/" + f_prefix + f1
            chunk_file = base_path + "input/" + f_prefix + f2
            vectors = __generate_embeddings(plain_file, chunk_file, chunk_embedding, idx, error_writer)
            vectors_combined.update(vectors)
        _write_vectors(vectors_combined, output_file)
