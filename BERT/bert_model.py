
import os
import torch
from torch import nn
from .packages import BertConfig, BertOutput
from .layers import LayerNorm as BertLayerNorm
from .bert_layer import BertLayer, BertEmbeddings

class BertEncoder(nn.Module):
    """BERT 模型中的Encoder部分,由多个BERTlayer组成"""

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        #存储所有的 hidden states和attentions
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):#i=0,1,2,3,4,5,6,7,8,9,10,11 ; layer_module=BertLayer
            """
            ModuleList(
                    (0-11): 12 x BertLayer(
                        (attention): BertAttention(
                        (self): MultiHeadAttentionLayer(
                            (query): Linear(in_features=768, out_features=768, bias=True)
                            (key): Linear(in_features=768, out_features=768, bias=True)
                            (value): Linear(in_features=768, out_features=768, bias=True)
                            (dropout): Dropout(p=0.1, inplace=False)
                        )
                        (output): BertAddNorm(
                            (dense): Linear(in_features=768, out_features=768, bias=True)
                            (layer_norm): LayerNorm()
                            (dropout): Dropout(p=0.1, inplace=False)
                        )
                        )
                        (intermediate): BertIntermediate(
                        (dense): Linear(in_features=768, out_features=3072, bias=True)
                        )
                        (output): BertAddNorm(
                        (dense): Linear(in_features=3072, out_features=768, bias=True)
                        (layer_norm): LayerNorm()
                        (dropout): Dropout(p=0.1, inplace=False)
                        )
                    )
                    )
            """
            if self.output_hidden_states:#隐藏层
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])#前向传播,这里进BERTLayer
            # [0] 是 embedding, [1] 是 attention_score
            hidden_states = layer_outputs[0]#更新

            if self.output_attentions:#注意力层
                all_attentions = all_attentions + (layer_outputs[1],)

       
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)

        if self.output_hidden_states:
            # 把中间层的结果取出来，一些研究认为中间层的 embedding 也有价值
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class BertPooler(nn.Module):
    """
    BERT模型中的池化层
    """
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 提取第一个标记（通常是[CLS]标记）的隐藏状态
        first_token_tensor = hidden_states[:, 0]

        #使用全连接层和激活函数生成池化表示
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT模型的主题,包含了BERTEmbeddings, BertEncoder, BertPooler三个部分"""
    def __init__(self, config_path):
        super(BertModel, self).__init__()

        self.config = BertConfig(os.path.join(config_path, "config.json"))

        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

        self.init_weights()
        self.from_pretrained(os.path.join(os.path.join(config_path, "pytorch_model.bin")))
        self.eval()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def from_pretrained(self, pretrained_model_path):
        if not os.path.exists(pretrained_model_path):
            print(f"missing pretrained_model_path: {pretrained_model_path}")
            pass

        state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # 名称可能存在不一致，进行替换
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = key
            if 'gamma' in key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
            # 兼容部分不优雅的变量命名
            if 'LayerNorm' in key:
                new_key = new_key.replace('LayerNorm', 'layer_norm')


            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):

            if new_key in self.state_dict().keys():
                state_dict[new_key] = state_dict.pop(old_key)
            else:
                # 避免预训练模型里有多余的结构，影响 strict load_state_dict
                state_dict.pop(old_key)

        # 加载权重到模型中
        self.load_state_dict(state_dict, strict=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        #创建扩展的注意力掩码
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)#[1,64] -> [1,1,64] -> [1,1,64,64]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0#mask


        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = BertOutput(last_hidden_state=sequence_output, pooler_output=pooled_output,
                             attentions=encoder_outputs[1:])

        return outputs
