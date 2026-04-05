import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from models.bidaf import BiDAF
from models.bert_bidaf import BertBiDAF
from data.squad_loader import SQuADBiDAFDataset
from utils.metrics import compute_em_f1

cfg = Config()

def evaluate(model_type='bidaf', checkpoint_path=None):
    if model_type == 'bidaf':
        dataset = SQuADBiDAFDataset(split='validation', use_bert=False)
        model = BiDAF(len(dataset.word2idx), len(dataset.char2idx)).to(cfg.device)
        collate_fn = collate_bidaf  # need to import from train_bidaf or redefine
        # For brevity, we reuse the collate function defined earlier.
        # In a real script, you would import from train_bidaf or write a separate collate.
    else:
        dataset = SQuADBiDAFDataset(split='validation', use_bert=True)
        model = BertBiDAF().to(cfg.device)
        collate_fn = collate_bert

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    model.eval()

    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate_fn)
    total_em = 0
    total_f1 = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            if model_type == 'bidaf':
                c_word, c_char, q_word, q_char, start_gt, end_gt = batch
                c_word, c_char, q_word, q_char = [x.to(cfg.device) for x in [c_word, c_char, q_word, q_char]]
                start_logits, end_logits = model(c_word, c_char, q_word, q_char)
            else:
                input_ids, attn_mask, token_type_ids, start_gt, end_gt = batch
                input_ids, attn_mask, token_type_ids = [x.to(cfg.device) for x in [input_ids, attn_mask, token_type_ids]]
                start_logits, end_logits = model(input_ids, attn_mask, token_type_ids)

            start_pred = torch.argmax(start_logits, dim=1)
            end_pred   = torch.argmax(end_logits, dim=1)
            # For simplicity, we compute EM/F1 only on position indices (not on actual text)
            # In a full evaluation you would decode the text and compare.
            # Here we assume ground truth and predicted spans are exact index matches.
            em = (start_pred == start_gt) & (end_pred == end_gt)
            # F1 on indices is not meaningful; this is a placeholder.
            # Replace with text decoding using the dataset's original context.
            total_em += em.sum().item()
            total += len(start_gt)

    print(f'EM: {total_em/total:.4f}')

if __name__ == '__main__':
    # Example: evaluate the best checkpoint
    evaluate('bidaf', checkpoint_path='./checkpoints/bidaf_epoch15.pt')