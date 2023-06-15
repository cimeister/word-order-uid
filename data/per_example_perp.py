# This script loads a trained fairseq language model and measures surprisal on a test set.
# It then saves the per-token surprisal and the tokens themselves to a file.
#
# Notes on adaptive language modelling.
# ------------------------------------
# For adaptive LM, set the following line in  fairseq/models/fairseq_model.py must be commented out!
# https://github.com/facebookresearch/fairseq/blob/b30980349bcb2e870481d783ac8cb3f338361601/fairseq/models/fairseq_model.py#L180
# Also note that the LM currently keeps adapting throughout the test set, while in reality it should
# only adapt on a single document at a time.

import argparse
import numpy
import random
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
from torch.optim import Adam
import tqdm

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
parser.add_argument("--n_docs", type=int)
args = parser.parse_args()


# Set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)

with open(args.test_file, "r") as f:
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

    lprobs = []
    count = 0
    perps = []
    tokens = []
    for l in tqdm(lines[: args.n_docs]):
        if custom_lm_hub.encode(l).size(0) > custom_lm_hub.max_positions - 2:
            l = " ".join(l.split()[: custom_lm_hub.max_positions - 2])
        out = custom_lm_hub.score(l, shorten_method="truncate")
        perps.append(out["positional_scores"].mean().neg().exp().item())
        lprobs.append(out["positional_scores"])
        tokens.append([custom_lm_hub.tgt_dict[i] for i in out["tokens"]])

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

    print(args.checkpoint_dir, perps)
    torch.save([lprobs, tokens], args.out_file)
