import torch

def padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2).to(torch.uint8)   # [B, 1, L]

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = 1- torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

def test():
    # 以最简化的形式测试Transformer的两种mask
    seq = torch.LongTensor([[1,2,0]]) # batch_size=1, seq_len=3，padding_idx=0
    embedding = torch.nn.Embedding(num_embeddings=3, embedding_dim=10, padding_idx=0)
    query, key = embedding(seq), embedding(seq)
    scores = torch.matmul(query, key.transpose(-2, -1))

    mask_p = padding_mask(seq, 0)
    mask_s = sequence_mask(seq)
    mask_decoder = mask_p & mask_s # 结合 padding mask 和 sequence mask
    scores_encoder = scores.masked_fill(mask_p==0, -1e9) # 对于scores，在mask==0的位置填充
    scores_decoder = scores.masked_fill(mask_decoder==0, -1e9)

test()