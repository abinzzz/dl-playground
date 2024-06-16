
import collections
import os
import unicodedata
from io import open

import numpy as np
import torch

from .packages import TokenizerOutput
from .logger import init_logger

logger = init_logger(__name__)#日记记录器


def truncate_sequences(maxlen, indices, *sequences):
    """
    截断总长度至不超过 maxlen

    参数:
    - maxlen: 最大长度
    - indices: 序列中需要截断的位置
    - sequences: 需要截断的多个序列

    返回:
    - 截断后的序列集合
    """
    sequences = [s for s in sequences if s]#过滤掉空的序列 (['[CLS]', '我', '爱', '玩', '，', '原', '神', '[SEP]'], None)
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]#获取每个序列的长度 8
        if sum(lengths) > maxlen:
            # 从较长的一侧进行 pop
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def load_vocab(vocab_file):
    """
    vocab -> dict
    """
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """
    去除空白符
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class Tokenizer(object):

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]"):
        """
        参数:
            vocab_file:词典文件
            do_lower_case:是否转换成小写
            do_basic_tokenize:是否在分词前进行基本的分词处理
            unk_token:未知词标记
            sep_token:句子切分标记
            pad_token:填充标记
            cls_token:分类标记
            mask_token:mask标记

        """
        if not os.path.isfile(vocab_file):#文件是否存在
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)#加载词典  OrderedDict([('[PAD]', 0), ('[unused1]', 1), ('[unused2]', 2), ...)
        self.ids_to_tokens = collections.OrderedDict( # OrderedDict([(0, '[PAD]'), (1, '[unused1]'), (2, '[unused2]'), ...)
            [(ids, tok) for tok, ids in self.vocab.items()])#映射：id -> token
        
        self.do_basic_tokenize = do_basic_tokenize#是否基本分词
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=(unk_token, sep_token, pad_token, cls_token, mask_token))
            
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)#WordPiece 分词器
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    def tokenize(self, text):
        """
        text -> tokens
        """
        split_tokens = []
        if self.do_basic_tokenize:#基本分词
            for token in self.basic_tokenizer.tokenize(text):#基本分词
                for sub_token in self.wordpiece_tokenizer.tokenize(token):#wordpiece分词
                    split_tokens.append(sub_token)
        else:#wordpiece分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        if self.cls_token is not None:
            split_tokens.insert(0, self.cls_token)#插入cls_token
        if self.sep_token is not None:
            split_tokens.append(self.sep_token)#插入sep_token
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """
        tokens 转为 vocab 中的 id
        """
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """
        ids -> tokens
        """
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def encode(
            self,
            first_text,
            second_text=None,
            is_padding=True,
            max_len=512,
            truncate_from='right'
    ):
        """
        文本 --编码--> token id + segment id
        """
        if isinstance(first_text, str):
            first_tokens = self.tokenize(first_text)

        else:#这里输入的就是tokens
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif isinstance(second_text, str):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if max_len is not None:
            if truncate_from == 'right':
                index = -2
            elif truncate_from == 'left':
                index = 1
            else:
                index = truncate_from
            if second_text is not None:
                max_len += 1
            truncate_sequences(max_len, index, first_tokens, second_tokens)

        # token_ids=input_ids , segment_ids=token_type_ids
        first_token_ids = self.convert_tokens_to_ids(first_tokens)#['[CLS]', '我', '爱', '玩', '，', '原', '神', '[SEP]'] -> [101, 2769, 4263, 2157, 8024, 1378, 4906, 102]
        first_segment_ids = [0] * len(first_token_ids)#[0, 0, 0, 0, 0, 0, 0, 0]

        if second_text is not None:#不止一句话
            second_tokens = second_tokens[1:]
            second_token_ids = self.convert_tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
            
        attention_mask = [1] * len(first_token_ids)# [1, 1, 1, 1, 1, 1, 1, 1]
        
        # 做padding
        if is_padding:
            while len(first_token_ids) < max_len:
                first_token_ids.append(self.vocab[self.pad_token])
                first_segment_ids.append(self.vocab[self.pad_token])
                attention_mask.append(0)
        
        #截断
        if max_len and len(first_token_ids) > max_len:
            first_token_ids = first_token_ids[:max_len]
            first_segment_ids = first_segment_ids[:max_len]
            attention_mask = attention_mask[:max_len]

        first_token_ids = torch.tensor([first_token_ids])#len=8 -> 64 [101, 2769, 4263, 2157, 8024, 1378, 4906, 102]后面补0
        first_segment_ids = torch.tensor([first_segment_ids])#len=8 -> 64 [0, 0, 0, 0, 0, 0, 0, 0]后面补0
        attention_mask = torch.tensor([attention_mask])#len=8 -> 64 [1, 1, 1, 1, 1, 1, 1, 1]后面补0

        return TokenizerOutput(first_token_ids, first_segment_ids, attention_mask)


class BasicTokenizer(object):
    """
    执行基本的分词操作(标点符号分割等)
    """

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """

        Args:
          do_lower_case: 转小写
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """
        text->tokens,可能会影响标注类任务位置的对齐

        """
        text = self._clean_text(text)#'我爱玩，原神' -> '我爱玩，原神'
        text = self._tokenize_chinese_chars(text)#'我爱玩，原神' -> '我 爱 玩 ， 原 神'
        orig_tokens = whitespace_tokenize(text)#'我 爱 玩 ， 原 神' -> ['我', '爱', '玩', '，', '原', '神']
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()#'我'
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))#['我']

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """去重音"""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """对标点符号进行分割"""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """中文字符附近添加空格"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """判断是否为中文字符"""

        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            
            return True

        return False

    def _clean_text(self, text):
        """移除无效和空白字符"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """WordPiece 分词器"""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        """
        参数:
          vocab: 词汇表。
          unk_token: 未知词标记。
          max_input_chars_per_word: 每个词的最大字符数。
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        text(基本分词处理后)->WordPiece tokens
        """

        output_tokens = []
        for token in whitespace_tokenize(text):#text='我'
            chars = list(token)#['我']
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """是否是空白字符"""

    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """是否是控制字符"""

    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """是否是标点符号"""
    cp = ord(char)

    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
