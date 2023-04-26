# This script loads a trained fairseq language model and evaluates it on a test set
# Command-line arguments:
# 1. directory with checkpoint. assumes the dir contains a checkpoint_best.pt
# 2. directory with test data. expects the result of fairseq-preprocess (bin data)
# 3. file with test data. expects plain text with bpe
# 4. output file where logprobs and tokens will be saved


import sys
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
import argparse
from scipy.stats import entropy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--test_file")
    parser.add_argument("--out_file")
    args = parser.parse_args()

    # checkpoint_dir = sys.argv[1]
    # data_dir = sys.argv[2]
    # test_file = sys.argv[3]
    # out_file = sys.argv[4]

    with open(args.test_file, "r") as f:
        lines = f.read().splitlines()
        custom_lm = TransformerLanguageModel.from_pretrained(
            args.checkpoint_dir,
            data_name_or_path=args.data_dir,
            checkpoint_file="checkpoint_best.pt",
        )
        surprisals = []
        entropies = []
        count = 0
        perps = []
        tokens = []
        for l in lines:

            # truncate if necessary
            if custom_lm.encode(l).size(0) > custom_lm.max_positions - 2:
                l = " ".join(l.split()[: custom_lm.max_positions - 2])

            # encode inputs and feed to model
            encoded = custom_lm.encode(l).unsqueeze(0)
            logprobs, extra = custom_lm.models[0](encoded)

            # INFO
            # expected logprobs shape: [batch_size (1), number of tokens in sentence, vocab size]

            # append tokens
            decoded = custom_lm.decode(encoded)
            tokens.append(decoded)

            # append surprisals
            # surprisals.append(logprobs.neg())
            out = custom_lm.score(l, shorten_method="truncate")
            surprisals.append(-1 * out["positional_scores"][-1])

            # append entropies
            ent = entropy(logprobs[0, :, :], axis=1)
            entropies.append(ent)

            # INFO
            # expected ent shape: [number of tokens in sentence]

            # top_k_ids = (
            #     logprobs[0, -2, :].argsort(descending=True)[: args.top_k].tolist()
            # )

            # out = custom_lm.score(l, shorten_method="truncate")
            # perps.append(out["positional_scores"].mean().neg().exp().item())
            # lprobs.append(out["positional_scores"])
            # tokens.append([custom_lm.tgt_dict[i] for i in out["tokens"]])

        print(args.checkpoint_dir, perps)
        torch.save([tokens, surprisals, entropies], args.out_file)

