import torch
from torch.autograd import Variable

from models.VAETransformer.mask import create_src_mask, nopeak_mask

# torch function:
# torch.max(input, dim): return (max, max_indices)
# torch.multinomial(input, num_samples): return (max_indices, num_samples)

# next time try beam search:
# ref1: https://zhuanlan.zhihu.com/p/339207092
# ref2: https://towardsdatascience.com/the-three-decoding-methods-for-nlp-23ca59cb1e9d

def decode(model, src, conds, max_len, type, use_cond2dec=False):
    src_mask = create_src_mask(src=src, cond=conds)

    z, _, _, _ = model.encode(src, conds, src_mask)

    # initialize the record for break condition. 0 for non-stop, while 1 for stop 
    break_condition = torch.zeros(src.shape[0], dtype=torch.bool)
    
    # create a batch of starting tokens (1)
    ys = torch.ones(src.shape[0], 1, requires_grad=True).type_as(src.data)

    for i in range(max_len-1):
        with torch.no_grad():
            # create a sequence (nopeak) mask for target
            # use_cond2dec should be true s.t. trg_mask considers both the conditions and smiles tokens
            trg_mask = nopeak_mask(ys.size(-1), conds.size(1), src.get_device(), use_cond2dec) 
            # dim. of output: (bs, ys.size(-1)+1, d_model)
            output = model.decode(ys, z, conds, src_mask, trg_mask)[0]
            # dim. of output: (bs, ys.size(-1)+1, vocab_size)
            output = model.generator(output)
            # we care about the last token
            output = output[:, -1, :]
            # may need to check if the probability of the output is log-based
            prob = torch.exp(output)

            if type == 'greedy':
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]
            elif type == 'multinomial':
                next_word = torch.multinomial(prob, 1) # shape: (batch_size, 1)
                ys = torch.cat([ys, next_word], dim=1) #[batch_size, i]
                next_word = torch.squeeze(next_word) # shape: (batch_size)
            
            # update the break condition. 2 is the stop token
            break_condition = (break_condition | (next_word.to('cpu')==2))
            # If all satisfies the break condition, then break the loop.
            if all(break_condition):
                break

    return ys