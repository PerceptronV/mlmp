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
    def __init__(
        self,
        data_dir: Path,
        include_support_programs: bool = True,
        include_query_program: bool = True,
    ):
        self.tokeniser = Tokeniser()
        self.data_dir = data_dir
        self.files = sorted(data_dir.glob('episode_*.json'))
        self.include_support_programs = include_support_programs
        self.include_query_program = include_query_program

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
    
    def tokenise_episode(
        self,
        episode: dict,
        n_io_shown: int,
        include_support_programs: bool = True,
        include_query_program: bool = True,
    ):
        x = []
        
        for ex in episode['support_examples']:
            for io in ex['io_pairs']:
                x.extend(self.tokeniser.tokenise_list(io['input']) + [self.to])
                x.extend(self.tokeniser.tokenise_list(io['output']) + [self.newline])
            if include_support_programs:
                x.extend(self.tokeniser.tokenise_program(ex['program_shuffled']) + [self.newline])
            x.extend([self.sep] + [self.newline])
        
        x.extend([self.sep] + [self.newline])
        
        for io in episode['query']['io_pairs'][:n_io_shown]:
            x.extend(self.tokeniser.tokenise_list(io['input']) + [self.to])
            x.extend(self.tokeniser.tokenise_list(io['output']) + [self.newline])
        
        if include_query_program:
            y = [self.start] + self.tokeniser.tokenise_program(episode['query']['program_shuffled']) + [self.end]
        else:
            y = [self.start]
            for io in episode['query']['io_pairs'][n_io_shown:]:
                x.extend(self.tokeniser.tokenise_list(io['input']) + [self.to] + [self.newline])
                y.extend(self.tokeniser.tokenise_list(io['output']) + [self.newline])
            y.append(self.end)

        return x, y
    
    def detokenise_episode(self, x: list[int], y: list[int]):
        return {
            'x': self.tokeniser.detokenise(x),
            'y': self.tokeniser.detokenise(y)
        }
    
    @property
    def max_n_io(self):
        return self.n_io if self.include_query_program else self.n_io - 1

    def __len__(self):
        return len(self.files) * self.max_n_io
    
    def __getitem__(self, idx, include_episode: bool = False):
        file_idx = idx // self.max_n_io
        n_io_shown = idx % self.max_n_io + 1
        episode = self.load_episode(self.files[file_idx])

        x, y = self.tokenise_episode(
            episode=episode,
            n_io_shown=n_io_shown,
            include_support_programs=self.include_support_programs,
            include_query_program=self.include_query_program,
        )
        # loss mask is has length of seq_len - 1 (you don't predict first token)
        # and is 1 at the last len(y) - 1 positions (the ones after <start>)
        loss_mask = [0] * len(x) + [1] * (len(y) - 1)
        
        if include_episode:
            episode['n_io_shown'] = n_io_shown
            return x + y, loss_mask, episode
        else:
            return x + y, loss_mask
    
    def compute_max_lengths(self, verbose: bool = False):
        if self.maxx is None or self.maxy is None or self.maxtotal is None:
            maxx = -1
            maxy = -1
            maxtotal = -1

            for i in tqdm(range(len(self.files)), desc="Computing max lengths", disable=not verbose):
                x, y = self.tokenise_episode(
                    episode=self.load_episode(self.files[i]),
                    n_io_shown=self.max_n_io,
                    include_support_programs=self.include_support_programs,
                    include_query_program=self.include_query_program,
                )
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

    while 1:

        try:
            dataset = input('Enter dataset directory (datasets/query_first_template_seed42_semvar/train): ').strip()
            if dataset == '':
                dataset = 'datasets/query_first_template_seed42_semvar/train'
            
            supp = input('Include support programs? (Y/n): ').lower()
            if supp == 'n':
                include_support_programs = False
            else:
                include_support_programs = True
            
            query = input('Include query program? (Y/n): ').lower()
            if query == 'n':
                include_query_program = False
            else:
                include_query_program = True
            
            dataset = ProgramDataset(
                data_dir=Path(dataset),
                include_support_programs=include_support_programs,
                include_query_program=include_query_program,
            )

            flag = input('Compute max lengths? (y/N): ').lower()
            if flag == 'y':
                print(f"Max lengths: {dataset.compute_max_lengths(verbose=True)}\n")

            while 1:
                idx = input('\nEnter index (-1 to exit): ')
                try:
                    idx = int(idx)
                except ValueError:
                    print('Invalid index. Please enter a valid integer.')
                    continue
                if idx < 0:
                    print('\n')
                    break
                if idx >= len(dataset):
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
    
        except KeyboardInterrupt:
            break
