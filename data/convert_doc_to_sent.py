import argparse
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile")
    parser.add_argument("--outputfile")
    parser.add_argument("--language")
    parser.add_argument("--partition")
    args = parser.parse_args()

    with open(args.inputfile) as f_in, open(args.outputfile, "w") as f_out:
        for doc in f_in:
            sentences = doc.split(".")
            sentences = [s for s in sentences if len(s.strip()) > 0]
            sentences = [s.strip() + " .\n" for s in sentences]
            f_out.writelines(sentences)

            # write a blank line if test partition to be able to separate documents
            if args.partition == "test":
                f_out.write("\n")

