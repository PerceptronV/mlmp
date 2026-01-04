"""Compare template vs hybrid vs empirical composers."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lang.composers import TemplateComposer, HybridComposer, EmpiricalComposer
from src.lang.grammar import DefaultGrammar
from src.lang.ast_nodes import pretty_print
from src.lang.type_utils import SubstitutionTable
from typing import Callable
from pathlib import Path
from collections import Counter

grammar = DefaultGrammar
functions_path = project_root / 'src' / 'data' / 'rule' / 'functions.txt'

target_type = Callable[[list[int]], list[int]]

def categorize(prog):
    ho_funcs = {'map', 'mapi', 'filter', 'filteri', 'sort', 'fold', 'foldi'}
    ho_count = sum(1 for fn in ho_funcs if f'({fn} ' in prog)

    has_simple_wrapper = any(f'({op} (' in prog for op in ['reverse', 'unique'])
    is_composition = ho_count >= 2 or has_simple_wrapper

    if prog == '(λ x x)':
        return 'identity'
    if is_composition:
        return 'composition'
    if '(mapi ' in prog:
        return 'mapi'
    if '(map ' in prog:
        return 'map'
    if '(filteri ' in prog:
        return 'filteri'
    if '(filter ' in prog:
        return 'filter'
    if '(sort ' in prog:
        return 'sort'
    if '(fold ' in prog:
        return 'fold'
    if '(if ' in prog:
        return 'conditional'
    return 'simple_op'

template_results = {'unique_set': set(), 'categories': Counter(), 'total': 0}
hybrid_results = {'unique_set': set(), 'categories': Counter(), 'total': 0}
empirical_results = {'unique_set': set(), 'categories': Counter(), 'total': 0}
empirical_no_arg_results = {'unique_set': set(), 'categories': Counter(), 'total': 0}

for seed in [42, 123, 456]:
    template = TemplateComposer(seed, grammar)
    hybrid = HybridComposer(seed, grammar, functions_path, noise=0.3)
    empirical = EmpiricalComposer(seed, grammar, functions_path, noise=0.3,
                                   include_function_name_in_context=True,
                                   include_arg_position_in_context=True)
    empirical_no_arg = EmpiricalComposer(seed + 1000, grammar, functions_path, noise=0.3,
                                          include_function_name_in_context=True,
                                          include_arg_position_in_context=False)

    for i in range(200):
        template.reset_var_counter()
        hybrid.reset_var_counter()
        empirical.reset_var_counter()
        empirical_no_arg.reset_var_counter()
        try:
            tp = template.generate(target_type, depth=3, context={}, substitutions=SubstitutionTable())
            tps = pretty_print(tp)
            if tps != '(λ x x)' and len(tps) >= 15:
                template_results['unique_set'].add(tps)
                template_results['categories'][categorize(tps)] += 1
                template_results['total'] += 1
        except:
            pass
        try:
            hp = hybrid.generate(target_type, depth=3, context={}, substitutions=SubstitutionTable())
            hps = pretty_print(hp)
            if hps != '(λ x x)' and len(hps) >= 15:
                hybrid_results['unique_set'].add(hps)
                hybrid_results['categories'][categorize(hps)] += 1
                hybrid_results['total'] += 1
        except:
            pass
        try:
            ep = empirical.generate(target_type, depth=3, context={}, substitutions=SubstitutionTable())
            eps = pretty_print(ep)
            if eps != '(λ x x)' and len(eps) >= 15:
                empirical_results['unique_set'].add(eps)
                empirical_results['categories'][categorize(eps)] += 1
                empirical_results['total'] += 1
        except:
            pass
        try:
            enp = empirical_no_arg.generate(target_type, depth=3, context={}, substitutions=SubstitutionTable())
            enps = pretty_print(enp)
            if enps != '(λ x x)' and len(enps) >= 15:
                empirical_no_arg_results['unique_set'].add(enps)
                empirical_no_arg_results['categories'][categorize(enps)] += 1
                empirical_no_arg_results['total'] += 1
        except:
            pass

print('=' * 100)
print('FINAL COMPARISON (600 samples each, across 3 seeds)')
print('=' * 100)
print()
print(f'{"Metric":<20} {"Template":>15} {"Hybrid":>15} {"Empirical":>15} {"Emp(no arg)":>15}')
print('-' * 82)
print(f'{"Total non-trivial":<20} {template_results["total"]:>15} {hybrid_results["total"]:>15} {empirical_results["total"]:>15} {empirical_no_arg_results["total"]:>15}')
print(f'{"Unique programs":<20} {len(template_results["unique_set"]):>15} {len(hybrid_results["unique_set"]):>15} {len(empirical_results["unique_set"]):>15} {len(empirical_no_arg_results["unique_set"]):>15}')
t_pct = 100*len(template_results["unique_set"])/max(1,template_results["total"])
h_pct = 100*len(hybrid_results["unique_set"])/max(1,hybrid_results["total"])
e_pct = 100*len(empirical_results["unique_set"])/max(1,empirical_results["total"])
en_pct = 100*len(empirical_no_arg_results["unique_set"])/max(1,empirical_no_arg_results["total"])
print(f'{"Uniqueness %":<20} {t_pct:>14.1f}% {h_pct:>14.1f}% {e_pct:>14.1f}% {en_pct:>14.1f}%')
print()
print('CATEGORY DISTRIBUTION:')
print(f'{"Category":<20} {"Template":>15} {"Hybrid":>15} {"Empirical":>15} {"Emp(no arg)":>15}')
print('-' * 82)
all_cats = sorted(set(template_results['categories'].keys()) | set(hybrid_results['categories'].keys()) |
                  set(empirical_results['categories'].keys()) | set(empirical_no_arg_results['categories'].keys()))
for cat in all_cats:
    tc = template_results['categories'][cat]
    hc = hybrid_results['categories'][cat]
    ec = empirical_results['categories'][cat]
    enc = empirical_no_arg_results['categories'][cat]
    print(f'{cat:<20} {tc:>15} {hc:>15} {ec:>15} {enc:>15}')

print()
print('CONCLUSION:')
results = [
    ('Template', len(template_results['unique_set']), template_results['categories'].get('composition', 0)),
    ('Hybrid', len(hybrid_results['unique_set']), hybrid_results['categories'].get('composition', 0)),
    ('Empirical', len(empirical_results['unique_set']), empirical_results['categories'].get('composition', 0)),
    ('Emp(no arg)', len(empirical_no_arg_results['unique_set']), empirical_no_arg_results['categories'].get('composition', 0)),
]
best_unique = max(results, key=lambda x: x[1])
best_comp = max(results, key=lambda x: x[2])
print(f'  Best for unique programs: {best_unique[0]} ({best_unique[1]} unique)')
print(f'  Best for compositions: {best_comp[0]} ({best_comp[2]} compositions)')
print()
print('  Empirical with arg_position vs without:')
e_unique = len(empirical_results['unique_set'])
en_unique = len(empirical_no_arg_results['unique_set'])
if e_unique > en_unique:
    print(f'    With arg_position: {e_unique - en_unique} MORE unique programs')
elif en_unique > e_unique:
    print(f'    Without arg_position: {en_unique - e_unique} MORE unique programs')
else:
    print(f'    Same number of unique programs')
