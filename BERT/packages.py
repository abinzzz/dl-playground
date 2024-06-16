# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# @author: mmmwhy <mmmwhy@mail.ustc.edu.cn>
# @date: 2022/01/25
#
""""""
import json
from typing import Optional, Tuple

import torch


class BertConfig:
    def __init__(self, vocab_size_or_config_json_file):
        with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value

    def __getattr__(self, key):
        """查找属性"""
        if key in self.__dict__:
            return self.__dict__[key]
        return None


class BertOutput:
    """保存 BERT 模型的输出"""
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __init__(self, last_hidden_state, pooler_output, attentions):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.attentions = attentions

class TokenizerOutput:
    """保存 BERT 模型的输入编码"""
    input_ids: torch.LongTensor = None
    token_type_ids: torch.LongTensor = None
    attention_mask: torch.LongTensor = None

    def __init__(self, token_ids, segment_ids, attention_mask):
        self.input_ids = token_ids
        self.token_type_ids = segment_ids
        self.attention_mask = attention_mask
