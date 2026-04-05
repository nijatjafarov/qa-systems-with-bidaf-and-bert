import os

class Config:
    # Paths
    data_dir = './data/squad'
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # Embedding dimensions
    glove_dim = 300
    char_dim = 64
    char_channel_width = 5
    char_channel_size = 100
    hidden_size = 128
    dropout = 0.2
    num_highway = 2

    # Training
    batch_size = 32
    epochs = 15
    lr = 0.001
    bert_lr = 3e-5
    max_seq_len = 384
    max_word_len = 16

    bert_name = 'bert-base-uncased'

    # Device
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'