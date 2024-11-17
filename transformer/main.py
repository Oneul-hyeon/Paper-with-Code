import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["make_tokenizer", "make_dataset", "train", "demo"])
args = parser.parse_args()
    
if args.mode == "make_tokenizer" :
    os.system("python src/make_tokenizer.py")
elif args.mode == "make_dataset" :
    os.system("python src/make_dataset.py")
elif args.mode == "train" :
    os.system("python src/train.py")
elif args.mode == "demo" :
    os.system("python src/demo.py")