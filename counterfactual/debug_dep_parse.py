# Generate dependency parses in CoNNL-U format for a given dataset
# Author: Thomas Hikaru Clark (thclark@mit.edu)
# Example Usage:
# python dep_parse.py --lang en --udpipe_model_path models/model.udpipe \
#   --data_dir wiki40b-txt-final --parse_dir wiki40b-txt-parsed \
#   --partitions "train,test,valid"

from ufal.udpipe import Model, Pipeline, ProcessingError
import sys
import argparse
import os
from mosestokenizer import (
    MosesPunctuationNormalizer,
    MosesTokenizer,
    MosesSentenceSplitter,
)

# from indicnlp.tokenize import sentence_tokenize, indic_tokenize
import indicnlp.tokenize
import hazm

# mapping from language code to preferred UDPipe model
UDPIPE_MODEL_LOOKUP = {
    "en": "udpipe_models/english-lines-ud-2.5-191206.udpipe",
    "ru": "udpipe_models/russian-syntagrus-ud-2.5-191206.udpipe",
    "de": "udpipe_models/german-hdt-ud-2.5-191206.udpipe",
    "fr": "udpipe_models/french-partut-ud-2.5-191206.udpipe",
    "vi": "udpipe_models/vietnamese-vtb-ud-2.5-191206.udpipe",
    "hi": "udpipe_models/hindi-hdtb-ud-2.5-191206.udpipe",
    "tr": "udpipe_models/turkish-imst-ud-2.5-191206.udpipe",
    "hu": "udpipe_models/hungarian-szeged-ud-2.5-191206.udpipe",
    "id": "udpipe_models/indonesian-gsd-ud-2.5-191206.udpipe",
    "fa": "udpipe_models/persian-seraji-ud-2.5-191206.udpipe",
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", help="2-letter language code such as en, ru, vi, etc.", default="en"
    )
    parser.add_argument(
        "--udpipe_model_path", help="path to UDPipe model file for this language"
    )
    parser.add_argument(
        "--data_dir",
        help="path to data directory with original (normal-order) text",
        default=".",
    )
    parser.add_argument(
        "--parse_dir",
        help="path to directory where CONLLU parses of sentences should be stored",
        default=".",
    )
    parser.add_argument(
        "--partitions",
        default="train,test,valid",
        help="comma-seprated list of partitions",
    )
    args = parser.parse_args()

    # create output directory if it doesn't yet exist
    if not os.path.exists(args.parse_dir):
        os.system(f"mkdir -p {args.parse_dir}")

    # load UDPipe Model
    sys.stderr.write("Loading model: ")
    if args.udpipe_model_path is None:
        model = Model.load(UDPIPE_MODEL_LOOKUP[args.lang])
        sys.stderr.write(f"{model}\n")
    else:
        model = Model.load(args.udpipe_model_path)
    if not model:
        sys.stderr.write(f"Cannot load model from file '{args.udpipe_model_path}'\n")
        sys.exit(1)
    sys.stderr.write("done\n")

    # create pipeline
    pipeline = Pipeline(
        model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
    )
    err = ProcessingError()

    # Make sentence tokenizer
    # sent_tokenize = MosesSentenceSplitter(args.lang)
    # word_tokenize = MosesTokenizer(args.lang, no_escape=True)
    normalize = MosesPunctuationNormalizer(args.lang)

    # iterate over partitions
    for partition in args.partitions.split(","):
        input_path = os.path.join(args.data_dir, f"{args.lang}.{partition}")
        output_path = os.path.join(args.parse_dir, f"{args.lang}.{partition}.conllu")

        with open(input_path) as f_in, open(output_path, "w") as f_out:

            # use iterator over lines in f_in to save memory
            for document in f_in:

                # Moses tokenizer will fail if the line is blank
                if (len(document.strip())) == 0:
                    sys.stderr.write("There was a blank line in the input file\n")
                    continue

                if args.lang == "fa":
                    sentences = hazm.sent_tokenize(document)
                    sentences_tokenized = [hazm.word_tokenize(s) for s in sentences]
                if args.lang == "hi":
                    # split sentences
                    sentences = indicnlp.tokenize.sentence_tokenize.sentence_split(
                        document, lang="hi"
                    )
                    # sentences_tokenized = [word_tokenize(normalize(s)) for s in sentences]
                    sentences_tokenized = [
                        indicnlp.tokenize.indic_tokenize.trivial_tokenize(s)
                        for s in sentences
                    ]
                sentences = [" ".join(s) for s in sentences_tokenized]
                sentences = "\n".join(sentences)

                print(sentences)

                # Process data
                processed = pipeline.process(sentences, err)
                if err.occurred():
                    sys.stderr.write(
                        f"An error occurred in run_udpipe: {err.message}\n"
                    )
                    sys.exit(1)

                f_out.write(processed)
