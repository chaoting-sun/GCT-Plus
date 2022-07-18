from SmilesPE.pretokenizer import atomwise_tokenizer
import re


class moltokenize(object):
    def tokenizer(self, sentence):
        return [tok for tok in atomwise_tokenizer(sentence) if tok != " "]

    @staticmethod
    def untokenizer(tokens, sos_idx, eos_idx, itos):
        smi = ""
        for token in tokens:
            if token == eos_idx:
                break
            elif token != sos_idx:
                smi += itos[token]
        return smi