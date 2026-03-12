import random

from ..lang.parser import parse
from ..lang.grammar import Grammar, DefaultGrammar
from ..lang.composers import get_composer
from ..lang.type_checker import TypeChecker
from ..lang.compiler import JITCompiler
from ..lang.type_utils import get_args


class GrammarShuffler:
    def __init__(
        self,
        seed: int,
        grammar: Grammar = DefaultGrammar,
        composer_name: str = 'random'
    ) -> None:
        self.grammar = grammar
        self.compiler = JITCompiler(grammar)
        self.checker = TypeChecker(grammar)
        self.composer = get_composer(composer_name, seed, grammar)
    
    def generate_basis(self):
        names = self.grammar.names
        types = self.grammar.function_types
        
        take_type = types[names.index('take')]
        print(take_type)
        ast = self.composer.generate(take_type, 3)
        print(ast)
        gen_type = self.checker.check(ast)
        print(gen_type)

        func = self.compiler.compile(ast)
        print(func(10, True))
        

if __name__ == '__main__':
    shuffler = GrammarShuffler(42)
    shuffler.generate_basis()
