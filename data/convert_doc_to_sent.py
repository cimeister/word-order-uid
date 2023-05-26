import argparse
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile")
    parser.add_argument("--outputfile")
    parser.add_argument("--language")
    args = parser.parse_args()

    with open(args.inputfile) as f_in, open(
        args.outputfile, "w"
    ) as f_out, MosesSentenceSplitter(args.language) as sent_split:
        for doc in f_in:
            sentences = [x + "\n" for x in sent_split([doc])]
            f_out.writelines(sentences)
            f_out.write("\n")

