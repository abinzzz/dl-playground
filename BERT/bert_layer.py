import math
import torch
from torch import nn
from .activate import activations
from .layers import LayerNorm as BertLayerNorm


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        """
        BERT 模型中的「input embedding」部分
        
        Args:
            vocab_size: 词汇表大小
            hidden_size: 内部神经网络的隐藏层大小
            type_vocab_size: 类型词汇表大小,一般为2,用于表示句子的第一句和第二句
            max_position_embeddings: 最大序列长度，用于生成位置编码
            hidden_dropout_prob: 隐藏层丢弃率
            layer_norm_eps: LayerNorm 层的 epsilon 参数
        """
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)#input_ids: tensor([[ 101, 2769, 4263, 4381, 8024, 1333, 4868,  102, 0,...] |  size[1,64]->[1,64,768]
        position_embeddings = self.position_embeddings(position_ids)#position_ids: tensor([[ 0,  1,  2,  3,  4,  5,  6,  7, ...,63] size[1,64]->[1,64,768]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)#type_ids: tensor([[0,0,...,0]]) size[1,64]->[1,64,768]

        # 按位相加
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config):
        """
        BERT 模型中的「Multi-Head Attention」部分
        
        Args:
            hidden_size: 隐藏层维度
            num_attention_heads: 注意力头的数量
            attention_probs_dropout_prob: 注意力概率的 dropout 概率
            attention_scale: 对 query 和 value 的乘积结果进行缩放，目的是为了稳定 softmax 结果
            return_attention_scores: 是否返回注意力矩阵
        """
        super(MultiHeadAttentionLayer, self).__init__()

        assert config.hidden_size % config.num_attention_heads == 0

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.return_attention_scores = config.return_attention_scores

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        调整张量的形状以适应多头注意力计算

        [batch_size, query_len, hidden_size] -> [batch_size, query_len, num_attention_heads, attention_head_size]

        hidden_size = num_attention_heads * attention_head_size
        """

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask=None, head_mask=None):
        """
        query shape: [batch_size, query_len, hidden_size]
        key shape: [batch_size, key_len, hidden_size]
        value shape: [batch_size, value_len, hidden_size]
        """

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)
        """
        mixed_query_layer shape: [batch_size, query_len, hidden_size]
        mixed_query_layer shape: [batch_size, key_len, hidden_size]
        mixed_query_layer shape: [batch_size, value_len, hidden_size]
        """

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        """
        query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]
        """

        #将K进行转置，然后 q 和 k 执行点积, 获得 attention score
        #attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention_mask: mask的地方是-inf, -inf --softmax--> 0
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对注意力结果进行 softmax， 得到 query 对于每个 value 的 score
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # 头掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask


        """
        value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]
        attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        
        key_len==value_len
        """
        context_layer = torch.matmul(attention_probs, value_layer)

        # context_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]

        # transpose、permute 等维度变换操作后，tensor 在内存中不再是连续存储的，而 view 操作要求 tensor 的内存连续存储，
        # 所以在调用 view 之前，需要 contiguous 来返回一个 contiguous copy；
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer shape: [batch_size, query_len, num_attention_heads, attention_head_size]

        # 注意这里又把最后两个纬度合回去了，做的是 view 操作
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer shape: [batch_size, query_len, hidden_size]
        

        # 是否返回attention scores, 注意这里是最原始的 attention_scores 没有归一化且没有 dropout
        # 第一个位置是产出的 embedding，第二个位置是 attention_probs，后边会有不同的判断
        outputs = (context_layer, attention_scores) if self.return_attention_scores else (context_layer,)
        return outputs


class BertAddNorm(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps):
        """
        BERT 模型中的「Add & Norm」部分

        功能详细说明：
        1、在 Multi-Head attention 后，所有的头注意力结果是直接 concat 在一起的(view 调整 size 也可以认为 concat 在一起)
            直接 concat 在一起的结果用起来也有点奇怪，所以需要有个 fc ，来帮助把这些分散注意力结果合并在一起；
        2、在 Feed Forward 操作后，纬度被提升到 intermediate_size，BertAddNorm 还实现了把纬度从 intermediate_size 降回 hidden_size 的功能；
        3、真正的 Add & Norm 部分，也就是 layer_norm(hidden_states + input_tensor) 这一行；

        Args:
            intermediate_size: 中间层的维度大小
            hidden_size: 隐藏层的维度大小
            hidden_dropout_prob: 隐藏层的 dropout 概率
            layer_norm_eps: LayerNorm 层的 epsilon 参数
        """
        super(BertAddNorm, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)#核心部分
        return hidden_states


class BertIntermediate(nn.Module):
    """
    BERT模型中的「Position-wise Feed-Forward Networks 」 的部分
    FFN(x) = max(0, xW1 + b1)W2 + b2

    这里只有 activate(xw1+b1) 的部分,没有外边的那个W2,因为在 BertAddNorm 里边放着
    """

    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = activations[hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        """
        BERT模型中的「Multi-Head Attention 和 Add & Norm」部分

        Args:
          hidden_size: 隐层纬度
          num_attention_heads: 注意力头的数量
          attention_probs_dropout_prob: attention prob 的 dropout 比例
          attention_scale: 对 query 和 value 的乘积结果进行缩放，目的是为了 softmax 结果稳定
          return_attention_scores: 是否返回 attention 矩阵
          hidden_dropout_prob: 隐层 dropout 比例
          layer_norm_eps: norm 下边的 eps
        """

        super(BertAttention, self).__init__()
        self.self = MultiHeadAttentionLayer(config)
        self.output = BertAddNorm(config.hidden_size, config.hidden_size,
                                  config.hidden_dropout_prob, config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, input_tensor, input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        """
        完整的 bert 单层结构
        """
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)

        self.intermediate = BertIntermediate(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.output = BertAddNorm(config.intermediate_size, config.hidden_size,
                                  config.hidden_dropout_prob, config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)#attention+AddNorm
        attention_output = attention_outputs[0]


        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        # attention_outputs[0] 是 embedding, [1] 是 attention_probs
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs
