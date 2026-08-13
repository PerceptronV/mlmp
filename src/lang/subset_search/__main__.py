"""CLI for the primitive subset search.

    python -m src.lang.subset_search --stage 0            # proxy sweep
    python -m src.lang.subset_search --stage 1 --top-n 200
    python -m src.lang.subset_search --stage 2

Type-restricted search (separate --out; stages 1-2 inherit the restriction):

    python -m src.lang.subset_search --stage 0 --out outputs/subset_search_ll \
        --target-type 'list[int]'
"""

import argparse

from . import search


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2])
    parser.add_argument('--out', default='outputs/subset_search')
    parser.add_argument('--max-size', type=int, default=None,
                        help='enumeration size bound (default: 7 stage 0, 10 stage 1)')
    parser.add_argument('--top-n', type=int, default=200)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max-finalists', type=int, default=10)
    parser.add_argument('--target-type', default=None, metavar='TYPE',
                        help="restrict the search to behaviors of this output "
                             "type, e.g. 'list[int]' for list[int]->list[int] "
                             "(stage 0 records it; stages 1-2 inherit it from "
                             "stage0.json)")
    args = parser.parse_args(argv)

    if args.stage == 0:
        scores = search.run_stage0(
            args.out, max_size=args.max_size if args.max_size is not None else 7,
            target_type=args.target_type,
        )
        print(f"Stage 0 done: {len(scores)} subsets scored; "
              f"top: {scores[0] if scores else 'n/a'}")
    elif args.stage == 1:
        results = search.run_stage1(
            args.out,
            top_n=args.top_n,
            max_size=args.max_size if args.max_size is not None else 10,
            timeout_s=args.timeout,
            workers=args.workers,
        )
        n_ok = sum(1 for r in results.values() if r['status'] == 'ok')
        print(f"Stage 1 done: {n_ok}/{len(results)} subsets scored ok")
    else:
        from .report import write_reports
        finalists = write_reports(args.out, max_finalists=args.max_finalists)
        print(f"Stage 2 done: {len(finalists)} finalist reports in "
              f"{args.out}/reports/")


if __name__ == '__main__':
    main()
