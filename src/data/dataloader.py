import json
from tqdm import tqdm
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
        self.files = sorted(data_dir.glob('episode_*.json'))

        assert len(self.files) > 0, f"No files found in {data_dir}"
        sample = self.load_episode(self.files[0])
        self.n_io = len(sample['query']['io_pairs'])

        self.pad = self.tokeniser.vocab.stoi[PAD_TOKEN]
        self.to = self.tokeniser.vocab.stoi[TO_TOKEN]
        self.newline = self.tokeniser.vocab.stoi[NEWLINE_TOKEN]
        self.sep = self.tokeniser.vocab.stoi[SEP_TOKEN]
        self.start = self.tokeniser.vocab.stoi[START_TOKEN]
        self.end = self.tokeniser.vocab.stoi[END_TOKEN]

        self.maxx = None
        self.maxy = None
    
    def load_episode(self, file: Path):
        with open(file, 'r') as f:
            return json.load(f)
    
    def tokenise_episode(self, episode: dict, n_io_shown: int):
        x = []
        for ex in episode['support_examples']:
            for io in ex['io_pairs']:
                x.extend(self.tokeniser.tokenise_list(io['input']) + [self.to])
                x.extend(self.tokeniser.tokenise_list(io['output']) + [self.newline])
            x.extend(self.tokeniser.tokenise_program(ex['program_shuffled']) + [self.newline] + [self.sep] + [self.newline])
        for io in episode['query']['io_pairs'][:n_io_shown]:
            x.extend(self.tokeniser.tokenise_list(io['input']) + [self.to])
            x.extend(self.tokeniser.tokenise_list(io['output']) + [self.newline])
        y = [self.start] + self.tokeniser.tokenise_program(episode['query']['program_shuffled']) + [self.end]
        return x, y
    
    def detokenise_episode(self, x: list[int], y: list[int]):
        return {
            'x': self.tokeniser.detokenise(x),
            'y': self.tokeniser.detokenise(y)
        }

    def __len__(self):
        return len(self.files) * self.n_io
    
    def __getitem__(self, idx):
        file_idx = idx // self.n_io
        n_io_shown = (idx % self.n_io) + 1
        x, y = self.tokenise_episode(self.load_episode(self.files[file_idx]), n_io_shown)
        # loss mask is has length of seq_len - 1 (you don't predict first token)
        # and is 1 at the last len(y) - 1 positions (the ones after <start>)
        loss_mask = [0] * len(x) + [1] * (len(y) - 1)
        return x + y, loss_mask
    
    def compute_max_lengths(self, verbose: bool = False):
        if self.maxx is None or self.maxy is None or self.maxtotal is None:
            maxx = -1
            maxy = -1
            maxtotal = -1

            for i in tqdm(range(len(self.files)), desc="Computing max lengths", disable=not verbose):
                x, y = self.tokenise_episode(self.load_episode(self.files[i]), self.n_io)
                maxx = max(maxx, len(x))
                maxy = max(maxy, len(y))
                maxtotal = max(maxtotal, len(x) + len(y))
        
            self.maxx = maxx
            self.maxy = maxy
            self.maxtotal = maxtotal
        
        return {
            'x': self.maxx,
            'y': self.maxy,
            'total': self.maxtotal,
        }
    
    @property
    def max_seq_len(self):
        if self.maxtotal is None:
            self.compute_max_lengths()
        return self.maxtotal


if __name__ == '__main__':
    dataset = input('Enter dataset directory (datasets/query_first_template_seed42/train): ').strip()
    if dataset == '':
        dataset = 'datasets/query_first_template_seed42/train'
    dataset = ProgramDataset(Path(dataset))

    flag = input('Compute max lengths? (y/N): ').lower()
    if flag == 'y':
        print(f"Max lengths: {dataset.compute_max_lengths(verbose=True)}\n")

    while 1:
        try:
            idx = int(input('Enter index (CTRL+C to exit): '))
        except KeyboardInterrupt:
            break
        except ValueError:
            print('Invalid index. Please enter a valid integer.')
            continue
        if idx < 0 or idx >= len(dataset):
            print(f'Index out of range [0, {len(dataset)-1}]. Please enter a valid index.')
            continue

        ep, loss_mask = dataset[idx]
        loss_mask = [0] + loss_mask # add 0 to the start to match the length of the sequence
        print(f"Length: {len(ep)}", end='\n ')
        for tok, mask in zip(ep, loss_mask):
            prtstr = dataset.tokeniser.vocab.itos[tok]
            if mask:
                prtstr = f'\033[92m{prtstr}\033[0m'
            print(prtstr, end=' ')
        print()
