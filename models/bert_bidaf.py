import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from config import Config

cfg = Config()

class BertBiDAF(nn.Module):
    def __init__(self, output_attentions=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.bert_name, output_attentions=output_attentions)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.bert_name)
        self.proj = nn.Linear(768, 2 * cfg.hidden_size)   # project to BiDAF's hidden size

        self.modeling_lstm = nn.LSTM(2*cfg.hidden_size, cfg.hidden_size, batch_first=True,
                                     bidirectional=True, dropout=cfg.dropout, num_layers=2)

        self.start_fc = nn.Linear(2*cfg.hidden_size, 1)
        self.end_fc   = nn.Linear(2*cfg.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT forward
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        bert_out = outputs.last_hidden_state          # (batch, seq_len, 768)

        # Project to BiDAF's hidden dimension
        H = self.proj(bert_out)                       # (batch, seq_len, 2*hidden)

        # Modeling layer (simplified BiDAF after attention)
        # In a true hybrid we would separate query/context and apply attention flow,
        # but here we use the fact that BERT already mixes them via token_type_ids.
        M, _ = self.modeling_lstm(H)

        start_logits = self.start_fc(M).squeeze(2)
        end_logits   = self.end_fc(M).squeeze(2)
        attentions = outputs.attentions
        return start_logits, end_logits, attentions