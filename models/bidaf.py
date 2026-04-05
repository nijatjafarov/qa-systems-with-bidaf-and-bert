import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

cfg = Config()

class CharacterEmbedding(nn.Module):
    def __init__(self, char_vocab_size):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, cfg.char_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(cfg.char_dim, cfg.char_channel_size,
                                cfg.char_channel_width, padding=cfg.char_channel_width//2)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, char_ids):
        # char_ids: (batch, seq_len, word_len)
        batch, seq_len, word_len = char_ids.size()
        char_emb = self.char_embed(char_ids)                       # (b,s,w,d_char)
        char_emb = char_emb.view(-1, word_len, cfg.char_dim)       # (b*s, w, d_char)
        char_emb = char_emb.transpose(1, 2)                        # (b*s, d_char, w)
        conv = self.conv1d(char_emb)                               # (b*s, C, w)
        conv = F.relu(conv)
        char_repr = F.max_pool1d(conv, conv.size(2)).squeeze(2)    # (b*s, C)
        char_repr = char_repr.view(batch, seq_len, -1)
        return self.dropout(char_repr)

class Highway(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.dropout(x)
        t = torch.sigmoid(self.gate(x))
        h = F.relu(self.fc(x))
        return t * h + (1 - t) * x

class BiDAF(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size):
        super().__init__()
        self.word_embed = nn.Embedding(word_vocab_size, cfg.glove_dim, padding_idx=0)
        self.char_embed = CharacterEmbedding(char_vocab_size)

        total_emb_dim = cfg.glove_dim + cfg.char_channel_size
        self.highway = Highway(total_emb_dim)

        self.contextual_lstm = nn.LSTM(total_emb_dim, cfg.hidden_size, batch_first=True,
                                       bidirectional=True, dropout=cfg.dropout)

        self.modeling_lstm = nn.LSTM(8*cfg.hidden_size, cfg.hidden_size, batch_first=True,
                                     bidirectional=True, dropout=cfg.dropout, num_layers=2)

        self.start_fc = nn.Linear(2*cfg.hidden_size, 1)
        self.end_fc   = nn.Linear(2*cfg.hidden_size, 1)

    def attend(self, H, U):
        # H: (batch, ctx_len, 2*hidden), U: (batch, q_len, 2*hidden)
        S = torch.bmm(H, U.transpose(1, 2))                       # (b, ctx, q)

        # Context-to-Query
        c2q_weights = F.softmax(S, dim=2)
        U_tilde = torch.bmm(c2q_weights, U)                       # (b, ctx, 2*h)

        # Query-to-Context
        q2c_weights = F.softmax(S.max(dim=2)[0], dim=1).unsqueeze(1)
        H_tilde = torch.bmm(q2c_weights, H)                       # (b, 1, 2*h)
        H_tilde = H_tilde.expand(-1, H.size(1), -1)

        G = torch.cat([H, U_tilde, H * U_tilde, H * H_tilde], dim=2)  # (b, ctx, 8*h)
        return G

    def forward(self, c_word, c_char, q_word, q_char):
        c_emb = torch.cat([self.word_embed(c_word), self.char_embed(c_char)], dim=2)
        q_emb = torch.cat([self.word_embed(q_word), self.char_embed(q_char)], dim=2)

        c_emb = self.highway(c_emb)
        q_emb = self.highway(q_emb)

        c_ctx, _ = self.contextual_lstm(c_emb)
        q_ctx, _ = self.contextual_lstm(q_emb)

        G = self.attend(c_ctx, q_ctx)

        M, _ = self.modeling_lstm(G)

        start_logits = self.start_fc(M).squeeze(2)
        end_logits   = self.end_fc(M).squeeze(2)
        return start_logits, end_logits