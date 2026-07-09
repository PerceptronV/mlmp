"""Tests for ProgramBank.all_programs() and the enumerator attempts counter."""

from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar


def _tiny_enum():
    g = pool_grammar(('+', 'length', 'take'))
    enum = BottomUpEnumerator(grammar=g, max_size=3)
    bank = enum.enumerate()
    return enum, bank


def test_all_programs_yields_every_stored_program():
    enum, bank = _tiny_enum()
    programs = list(bank.all_programs())
    assert len(programs) == bank.count()
    assert len(programs) > 0


def test_all_programs_unique_per_type_and_fingerprint():
    _, bank = _tiny_enum()
    seen = set()
    for p in bank.all_programs():
        if p.fingerprint is None:
            continue
        key = (str(p.type), p.fingerprint)
        assert key not in seen, f"duplicate behavior: {p.ast}"
        seen.add(key)


def test_attempts_counts_at_least_stored_programs():
    enum, bank = _tiny_enum()
    assert enum.attempts >= bank.count()
    assert enum.attempts > 0
