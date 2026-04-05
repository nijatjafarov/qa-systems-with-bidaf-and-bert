import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from models.bidaf import BiDAF
from data.squad_loader import SQuADBiDAFDataset

cfg = Config()

def collate_bidaf(batch):
    # Pad sequences to max length in batch
    c_word, c_char, q_word, q_char, s, e = zip(*batch)
    max_c_len = max(len(x) for x in c_word)
    max_q_len = max(len(x) for x in q_word)
    max_char_len = max(max(len(ch) for ch in seq) for seq in c_char + q_char)

    def pad_word(seq, max_len):
        return [s + [0]*(max_len - len(s)) for s in seq]

    def pad_char(seq, max_word_len, max_seq_len):
        padded = []
        for sent in seq:
            sent_pad = [w + [0]*(max_word_len - len(w)) for w in sent]
            sent_pad += [[0]*max_word_len] * (max_seq_len - len(sent))
            padded.append(sent_pad)
        return padded

    c_word = torch.tensor(pad_word(c_word, max_c_len))
    q_word = torch.tensor(pad_word(q_word, max_q_len))
    c_char = torch.tensor(pad_char(c_char, max_char_len, max_c_len))
    q_char = torch.tensor(pad_char(q_char, max_char_len, max_q_len))
    start = torch.tensor(s)
    end   = torch.tensor(e)
    return c_word, c_char, q_word, q_char, start, end

def train():
    dataset = SQuADBiDAFDataset(split='train', use_bert=False)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_bidaf)

    model = BiDAF(len(dataset.word2idx), len(dataset.char2idx)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}')
        for c_word, c_char, q_word, q_char, start, end in pbar:
            c_word, c_char, q_word, q_char = [x.to(cfg.device) for x in [c_word, c_char, q_word, q_char]]
            start, end = start.to(cfg.device), end.to(cfg.device)

            start_logits, end_logits = model(c_word, c_char, q_word, q_char)
            loss = criterion(start_logits, start) + criterion(end_logits, end)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), f'{cfg.save_dir}/bidaf_epoch{epoch+1}.pt')

if __name__ == '__main__':
    train()