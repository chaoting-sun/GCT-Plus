import torch
from torch.autograd import Variable

from models.cvae_Transformer.subsequent_mask import subsequent_mask

# torch function:
# torch.max(input, dim): return (max, max_indices)
# torch.multinomial(input, num_samples): return (max_indices, num_samples)

# next time try beam search:
# ref1: https://zhuanlan.zhihu.com/p/339207092
# ref2: https://towardsdatascience.com/the-three-decoding-methods-for-nlp-23ca59cb1e9d

def decode(model, src, src_mask, max_len, type):
    ys = torch.ones(src.shape[0], 1).type_as(src.data) # 1 is the starting symbol. shape of ys: (batch_size, 1)
    
    memory = model.encode(src, src_mask)
    break_condition = torch.zeros(src.shape[0], dtype=torch.bool)                          
    
    for i in range(max_len-1):
        with torch.no_grad():
            # out (batch_size, 1, d_model) -> (batch_size, max_len-1, d_model)
            out = model.decode(memory, src_mask, Variable(ys),
                               Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
            # take the last prediction of out
            log_prob = model.generator(out[:, -1, :])
            prob = torch.exp(log_prob)
           
            if type == 'greedy':
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]
            elif type == 'multinomial':
                next_word = torch.multinomial(prob, 1) # shape: (batch_size, 1)
                ys = torch.cat([ys, next_word], dim=1) #[batch_size, i]
                next_word = torch.squeeze(next_word) # shape: (batch_size)
            
            break_condition = (break_condition | (next_word.to('cpu')==2)) # 2: the stop token

            if all(break_condition): # all predictions end at 2
                break

    return ys