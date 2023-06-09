# This script loads a trained fairseq language model and evaluates it on a test set
# Command-line arguments:
# 1. directory with checkpoint. assumes the dir contains a checkpoint_best.pt
# 2. directory with test data. expects the result of fairseq-preprocess (bin data)
# 3. file with test data. expects plain text with bpe
# 4. output file where logprobs and tokens will be saved


import sys
import argparse
import numpy
import random
import torch
from ..fairseq.models.transformer_lm import TransformerLanguageModel
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter
from torch.optim import Adam

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_dir",
    help="Directory with checkpoint. Adsumes the dir contains a checkpoint_best.pt.",
)
parser.add_argument(
    "--data_dir",
    help="Directory with test data. Expects the result of fairseq-preprocess (bin data).",
)
parser.add_argument(
    "--test_file", help="File with test data. Expects plain text with bpe."
)
parser.add_argument(
    "--out_file", help="Output file where logprobs and tokens will be saved."
)
parser.add_argument(
    "--adapt_lr",
    help="Learning rate for each step of the adaptive LM; LM is not adaptive if this is not set.",
    type=float,
    default=None,
)
parser.add_argument(
    "--seed", help="Random seed for reproducibility", type=int, default=0
)
parser.add_argument(
    "--lang", help="ISO code of language, used for instantiating MosesSentenceSplitter"
)
args = parser.parse_args()

# Set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)

with open(args.test_file, "r") as f, MosesSentenceSplitter(args.lang) as sent_split:
    lines = f.read().splitlines()
    custom_lm_hub = TransformerLanguageModel.from_pretrained(
        args.checkpoint_dir,
        data_name_or_path=args.data_dir,
        checkpoint_file="checkpoint_best.pt",
    )
    if args.adapt_lr:
        print("Adaptive LM. Learning rate: ", args.adapt_lr)
        custom_lm = custom_lm_hub.models[0]
        optimizer = Adam(params=custom_lm.parameters(), lr=args.adapt_lr,)
        custom_lm.eval()

    lprobs, tokens = [], []
    count = 0
    perps = []
    lprobs_buff, tokens_buff = [], []

    # each line is a document, composed of one or more sentences
    for l in lines:
        if len(l) == 0:
            lprobs.append(lprobs_buff)
            tokens.append(tokens_buff)
            lprobs_buff.clear()
            tokens_buff.clear()
            continue
        for sentence in sent_split(l):
            if custom_lm_hub.encode(sentence).size(0) > custom_lm_hub.max_positions - 2:
                sentence = " ".join(sentence.split()[: custom_lm_hub.max_positions - 2])
            out = custom_lm_hub.score(sentence, shorten_method="truncate")
            perps.append(out["positional_scores"].mean().neg().exp().item())
            lprobs_buff.append(out["positional_scores"])
            tokens_buff.append([custom_lm_hub.tgt_dict[i] for i in out["tokens"]])

            if args.adapt_lr:
                custom_lm.train()
                prev_output_tokens = torch.tensor(
                    [custom_lm_hub.tgt_dict.eos()] + out["tokens"].tolist()
                )
                custom_lm.zero_grad()
                logits, _ = custom_lm(
                    prev_output_tokens.unsqueeze(0), return_all_hiddens=False
                )
                logp = custom_lm.get_normalized_probs(logits, log_probs=True)
                loss = -logp[range(out["tokens"].size(0)), out["tokens"]].mean()
                loss.backward()
                optimizer.step()
                custom_lm.eval()

    if lprobs_buff:
        lprobs.append(lprobs_buff)
        tokens.append(tokens_buff)
    # print(checkpoint_dir, perps)
    torch.save([lprobs, tokens], args.out_file)

