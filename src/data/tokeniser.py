from collections import Counter
from torchtext.vocab import Vocab

from ..lang.grammar import Grammar, DefaultGrammar
from ..lang.lexer import tokenise

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
TO_TOKEN = '→'
DEFINED_AS_TOKEN = '≜'
SEP_TOKEN = '========================================'
NEWLINE_TOKEN = '\n'


def get_vocab(grammar: Grammar = DefaultGrammar, int_max: int = 99, n_vars_max: int = 26):
    tok_specials = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN, TO_TOKEN, DEFINED_AS_TOKEN, SEP_TOKEN, NEWLINE_TOKEN]
    tokens = list(grammar.special_chars)
    tokens.extend(grammar.keywords)
    tokens.extend(grammar.names)
    tokens.extend(str(i) for i in range(int_max + 1))
    tokens.extend(f'_p{i}' for i in range(n_vars_max))

    vocab = Vocab(Counter({token: 1 for token in tokens}), specials=tok_specials)
    vocab.unk_index = vocab[UNK_TOKEN]
    return vocab


class Tokeniser:
    def __init__(self, grammar: Grammar = DefaultGrammar, int_max: int = 99, n_vars_max: int = 26):
        self.vocab = get_vocab(grammar, int_max, n_vars_max)
        self.int_max = int_max
        self.n_vars_max = n_vars_max
        self.grammar = grammar
    
    def tokenise_int(self, i: int) -> int:
        equiv = i % (self.int_max + 1)
        return self.vocab.stoi[str(equiv)]
    
    def tokenise_element(self, c: str) -> int:
        if c.isnumeric():
            return self.tokenise_int(int(c))
        return self.vocab.stoi[c]

    def tokenise_program(self, text: str, name_map: dict[str, str] | None = None) -> list[int]:
        """Tokenise a program. If ``name_map`` is given, every lexer ident token
        whose value is a key in ``name_map`` is rewritten to the mapped name
        before being looked up in the vocab. Used by the symbol-shuffling mode
        to substitute grammar function names per-episode."""
        toks = [t.value for t in tokenise(text) if t.value]
        if name_map:
            toks = [name_map.get(tok, tok) for tok in toks]
        return [self.tokenise_element(tok) for tok in toks]
    
    def tokenise_list(self, arr: list[int]) -> list[int]:
        return [self.vocab.stoi['[']] + \
               [self.tokenise_int(i) for i in arr] + \
               [self.vocab.stoi[']']]
    
    def detokenise(self, toks: list[int]) -> str:
        return ' '.join(self.vocab.itos[tok] for tok in toks)
