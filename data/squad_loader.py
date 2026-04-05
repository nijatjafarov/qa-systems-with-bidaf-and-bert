import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from config import Config
import numpy as np

cfg = Config()

class SQuADBiDAFDataset(Dataset):
    def __init__(self, split='train', use_bert=False):
        self.use_bert = use_bert
        self.data = load_dataset('squad', split=split)

        if not use_bert:
            # Build word and character vocabularies
            self.word2idx = {'<PAD>': 0, '<UNK>': 1}
            self.char2idx = {'<PAD>': 0, '<UNK>': 1}
            self._build_vocab()
        else:
            self.tokenizer = BertTokenizer.from_pretrained(cfg.bert_name, model_max_length=cfg.max_seq_len)

        self.examples = []
        for ex in self.data:
            context = ex['context']
            question = ex['question']
            answers = ex['answers']
            if not answers:
                continue
            answer_text = answers['text'][0]
            answer_start = answers['answer_start'][0]
            answer_end = answer_start + len(answer_text) - 1

            if not use_bert:
                # Convert to word and character indices
                c_tokens, c_word_ids, c_char_ids = self._tokenize_context(context)
                q_tokens, q_word_ids, q_char_ids = self._tokenize_question(question)

                # Map answer start/end to token positions
                start_pos, end_pos = self._find_answer_span(context, answer_start, answer_end, c_tokens)
                if start_pos is None:
                    continue
                self.examples.append((c_word_ids, c_char_ids, q_word_ids, q_char_ids, start_pos, end_pos))
            else:
                # BERT encoding
                encoded = self.tokenizer(
                    question, context,
                    max_length=cfg.max_seq_len,
                    truncation='only_second',   # keep question, truncate context
                    padding='max_length',
                    return_tensors='pt'
                )
                # Map answer start/end to token positions
                start_pos, end_pos = self._bert_answer_span(context, answer_start, answer_end, encoded)
                if start_pos is None:
                    continue
                self.examples.append((encoded['input_ids'].squeeze(0),
                                      encoded['attention_mask'].squeeze(0),
                                      encoded['token_type_ids'].squeeze(0),
                                      start_pos, end_pos))

    def _build_vocab(self):
        # Simple vocabulary from all words and characters in SQuAD
        for ex in self.data:
            for word in ex['context'].split() + ex['question'].split():
                word = word.lower()
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                for ch in word:
                    if ch not in self.char2idx:
                        self.char2idx[ch] = len(self.char2idx)

    def _tokenize_context(self, context):
        words = context.lower().split()
        word_ids = [self.word2idx.get(w, 1) for w in words]
        char_ids = [[self.char2idx.get(ch, 1) for ch in w[:cfg.max_word_len]] for w in words]
        # Pad char sequences
        max_wlen = max(len(c) for c in char_ids) if char_ids else 0
        char_ids = [c + [0]*(max_wlen - len(c)) for c in char_ids]
        return words, word_ids, char_ids

    def _tokenize_question(self, question):
        words = question.lower().split()
        word_ids = [self.word2idx.get(w, 1) for w in words]
        char_ids = [[self.char2idx.get(ch, 1) for ch in w[:cfg.max_word_len]] for w in words]
        max_wlen = max(len(c) for c in char_ids) if char_ids else 0
        char_ids = [c + [0]*(max_wlen - len(c)) for c in char_ids]
        return words, word_ids, char_ids

    def _find_answer_span(self, context, start_char, end_char, context_tokens):
        # Approximate mapping from character span to token indices
        pos = 0
        token_starts = []
        token_ends = []
        for token in context_tokens:
            token_starts.append(pos)
            pos += len(token) + 1  # +1 for space
            token_ends.append(pos - 2)  # last char index of token
        start_idx = None
        end_idx = None
        for i, (s, e) in enumerate(zip(token_starts, token_ends)):
            if s <= start_char <= e:
                start_idx = i
            if s <= end_char <= e:
                end_idx = i
        if start_idx is None or end_idx is None:
            return None, None
        return start_idx, end_idx

    def _bert_answer_span(self, context, start_char, end_char, encoded):
        # Use tokenizer's char_to_token method
        input_ids = encoded['input_ids'][0]
        offset_mapping = self.tokenizer(context, return_offsets_mapping=True)['offset_mapping']
        # The encoded sequence is [CLS] question [SEP] context [SEP], so we need to align
        # Simplified: only works if context starts at a known offset. For production use
        # the tokenizer's `return_offsets_mapping=True` on the full sequence.
        # Here we assume we have a method to map. For brevity, we return a placeholder.
        # In practice, use HuggingFace `tokenizer(..., return_offsets_mapping=True)`
        # and find the indices where offset_mapping overlaps with (start_char, end_char).
        start_pos = None
        end_pos = None
        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            if offset_start <= start_char < offset_end:
                start_pos = i
            if offset_start <= end_char < offset_end:
                end_pos = i
        return start_pos, end_pos

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]