import json, os, time, sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.enumeration.enumerator import BottomUpEnumerator

start = time.time()
enumerator = BottomUpEnumerator(max_size=8)
bank = enumerator.enumerate()
corpus = enumerator.extract_corpus()
elapsed = time.time() - start
print(f'Enumeration: {elapsed:.1f}s')

# Filter to list[int] -> list[int] programs
programs = [p for p in corpus if p.type == list[int]]
# Sort by size then by s-expression for determinism
programs.sort(key=lambda p: (p.size, str(p.ast)))
print(f'list[int] -> list[int] programs: {len(programs)}')

# Save in batches of 10000
out_dir = 'datasets/enumerated_list_int'
os.makedirs(out_dir, exist_ok=True)

batch_size = 10000
n_batches = (len(programs) + batch_size - 1) // batch_size

for i in range(n_batches):
    batch = programs[i * batch_size : (i + 1) * batch_size]
    entries = []
    for p in batch:
        entries.append({
            'program': str(p.ast),
            'size': p.size,
        })
    path = os.path.join(out_dir, f'batch_{i:04d}.json')
    with open(path, 'w') as f:
        json.dump(entries, f, indent=2)

print(f'Saved {n_batches} batches to {out_dir}/')

# Also save a manifest
manifest = {
    'total_programs': len(programs),
    'batch_size': batch_size,
    'n_batches': n_batches,
    'max_enum_size': 8,
    'type': 'list[int] -> list[int]',
    'programs_by_size': {},
}
from collections import Counter
size_counts = Counter(p.size for p in programs)
for s in sorted(size_counts):
    manifest['programs_by_size'][s] = size_counts[s]

with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Manifest written')
print(f'Programs by size:')
for s in sorted(size_counts):
    print(f'  size {s}: {size_counts[s]}')
