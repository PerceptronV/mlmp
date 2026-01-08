import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from ..data.tokeniser import (
    Tokeniser,
    PAD_TOKEN,
    UNK_TOKEN,
    START_TOKEN,
    END_TOKEN,
    TO_TOKEN,
    SEP_TOKEN,
    NEWLINE_TOKEN,
)


class ProgramDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.tokeniser = Tokeniser()
        self.data_dir = data_dir
        self.files = list(data_dir.glob('*.json'))

        assert len(self.files) > 0, f"No files found in {data_dir}"
        sample = self.load_episode(self.files[0])
        self.n_io = len(sample['query']['io_pairs'])

        self.to = self.tokeniser.vocab.stoi[TO_TOKEN]
        self.newline = self.tokeniser.vocab.stoi[NEWLINE_TOKEN]
        self.sep = self.tokeniser.vocab.stoi[SEP_TOKEN]
        self.start = self.tokeniser.vocab.stoi[START_TOKEN]
        self.end = self.tokeniser.vocab.stoi[END_TOKEN]
    
    def load_episode(self, file: Path):
        with open(file, 'r') as f:
            return json.load(f)
    
    def tokenise_episode(self, episode: dict):
        x = []
        for ex in episode['support_examples']:
            for io in ex['io_pairs']:
                x.extend(self.tokeniser.tokenise_list(io['input']) + [self.to])
                x.extend(self.tokeniser.tokenise_list(io['output']) + [self.newline])
            x.extend(self.tokeniser.tokenise_program(ex['program_canonical']) + [self.newline] + [self.sep] + [self.newline])
        for io in episode['query']['io_pairs']:
            x.extend(self.tokeniser.tokenise_list(io['input']) + [self.to])
            x.extend(self.tokeniser.tokenise_list(io['output']) + [self.newline])
        y = [self.start] + self.tokeniser.tokenise_program(episode['query']['program_canonical']) + [self.end]
        return x, y
    
    def detokenise_episode(self, x: list[int], y: list[int]):
        return {
            'x': self.tokeniser.detokenise(x),
            'y': self.tokeniser.detokenise(y)
        }

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.tokenise_episode(self.load_episode(self.files[idx]))


if __name__ == '__main__':
    idx = int(input('Enter index: '))
    dataset = ProgramDataset(Path('datasets/template_seed42/train'))
    ep = dataset[idx]
    print(dataset.detokenise_episode(ep[0], ep[1])['x'])
    print(dataset.detokenise_episode(ep[0], ep[1])['y'])
