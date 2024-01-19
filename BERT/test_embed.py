"""
- **Token Embedding**：将词元转换成固定维度的向量。
- **Segment Embedding**：用于句子对任务，区分两个句子。
- **Position Embedding**：反映词元在序列中的位置信息。
"""

# 定义函数 get_tokens_and_segments
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

# 示例输入
tokens_a = ["hello", "world"]
tokens_b = ["goodbye", "moon"]

# 调用函数并打印结果
tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
print(tokens)
print(segments)

"""
['<cls>', 'hello', 'world', '<sep>', 'goodbye', 'moon', '<sep>']
[0, 0, 0, 0, 1, 1, 1]
"""

