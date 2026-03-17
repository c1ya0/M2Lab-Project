from core.from_megamolbart.regex_tokenizer import RegExTokenizer
import torch
from typing import List

class REGRegExTokenizer(RegExTokenizer):
    def __init__(self):
        super().__init__()
        self.load_tokenizer()

        self.reg_token = '<REG>'
        self.vocab[self.reg_token] = 6
        self._update_cache()
        self._compile_regex()

        print(f"mask: {self.mask_id}, sep: {self.sep_id}, reg: {self.reg_id}")

    @property
    def reg_id(self):
        return 6

    def tokenize(self, smis: List[str]):
        tokens = [self.text_to_tokens(s) for s in smis]
        # Prepend <REG> token
        token_ids = [self.token_to_ids(['<REG>']+t) for t in tokens]
        pad_length = max([len(seq) for seq in token_ids])

        encoder_masks = [
            ([1] * len(seq)) + ([0] * (pad_length - len(seq)))
            for seq in token_ids
        ]
        token_ids = [
            seq + ([self.pad_id] * (pad_length - len(seq)))
            for seq in token_ids
        ]

        token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()
        encoder_masks = torch.tensor(encoder_masks,
                                     dtype=torch.int64,
                                     device=token_ids.device)

        return token_ids, encoder_masks

    def tokenize_pair(self, solu_smi: List[str], solv_smi: List[str]):
        solu_t = [self.text_to_tokens(s) for s in solu_smi]
        solv_t = [self.text_to_tokens(s) for s in solv_smi]

        # Prepend <REG> token, add <SEP> token between solu and solv
        token_ids = [self.token_to_ids(
            ['<REG>'] + solu + ['<SEP>'] + solv
        ) for solu, solv in zip(solu_t, solv_t)]

        pad_length = max([len(seq) for seq in token_ids])
        encoder_masks = [([1] * len(seq)) + ([0] * (pad_length - len(seq)))
                         for seq in token_ids]
        token_ids = [seq + ([self.pad_id] * (pad_length - len(seq)))
                     for seq in token_ids]

        token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()
        encoder_masks = torch.tensor(encoder_masks,
                                     dtype=torch.int64,
                                     device=token_ids.device)

        return token_ids, encoder_masks