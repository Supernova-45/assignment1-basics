import pickle

with open("output/output_owt_bpe/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

print("len(vocab) =", len(vocab))