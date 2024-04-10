from datasets import load_dataset, load_from_disk
#dataset = load_dataset("Salesforce/dialogstudio", "TweetSumm")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
dataset.save_to_disk("dataset1/wikitext") # 保存到该目录下
