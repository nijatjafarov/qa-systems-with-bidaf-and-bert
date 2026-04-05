import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from config import Config
from models.bert_bidaf import BertBiDAF
from data.squad_loader import SQuADBiDAFDataset

cfg = Config()

def collate_bert(batch):
    input_ids, attn_mask, token_type_ids, start, end = zip(*batch)
    input_ids = torch.stack(input_ids)
    attn_mask = torch.stack(attn_mask)
    token_type_ids = torch.stack(token_type_ids)
    start = torch.tensor(start)
    end   = torch.tensor(end)
    return input_ids, attn_mask, token_type_ids, start, end

def train():
    dataset = SQuADBiDAFDataset(split='train', use_bert=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_bert)

    model = BertBiDAF().to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.bert_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):  # BERT fine‑tuning needs fewer epochs
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}')
        for input_ids, attn_mask, token_type_ids, start, end in pbar:
            input_ids, attn_mask, token_type_ids = [x.to(cfg.device) for x in [input_ids, attn_mask, token_type_ids]]
            start, end = start.to(cfg.device), end.to(cfg.device)

            start_logits, end_logits = model(input_ids, attn_mask, token_type_ids)
            loss = criterion(start_logits, start) + criterion(end_logits, end)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), f'{cfg.save_dir}/bert_bidaf_epoch{epoch+1}.pt')

if __name__ == '__main__':
    train()