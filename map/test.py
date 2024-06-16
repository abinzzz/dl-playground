from datasets import load_dataset
ds = load_dataset("rotten_tomatoes", split="validation")

def add_prefix(example):
    """增加前缀"""
    example["text"] = "Review: " + example["text"]
    return example

ds = ds.map(add_prefix)

##查看修改后的样本
print(ds[0:3]["text"])


ds = ds.map(add_prefix, num_proc=4)