import torch
from torch import nn
from d2l import torch as d2l


#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)#10000, 768
        self.segment_embedding = nn.Embedding(2, num_hiddens)#2, 768
        self.blks = nn.Sequential()
        for i in range(num_layers):#transformer block
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）\
        
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        print("tokens:",tokens.shape)
        print("self.token_embedding(tokens):", self.token_embedding(tokens).shape)
        print("segments:",segments.shape)
        print("self.segment_embedding(segments):", self.segment_embedding(segments).shape)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        print("self.pos_embedding.data[:, :X.shape[1], :]:", self.pos_embedding.data[:, :X.shape[1], :].shape)
        
        #把embedding输入到block中
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
    
#词汇表大小、隐藏单元数、FFN隐藏层的隐藏单元数、多头注意力头数
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4

#layer norm shape，ffn输入层维度，transformer block个数，dropout
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)


tokens = torch.randint(0, vocab_size, (2, 8)) # (batch_size, max_len)
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])#(batch_size, max_len)
encoded_X = encoder(tokens, segments, None)#torch.Size([batch_size, max_len, num_hiddens])
#print(encoded_X)
print(encoded_X.shape)
