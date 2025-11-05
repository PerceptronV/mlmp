"""
Backend detection and information for lang_opt.

This module provides utilities to check if the C++ backend is available
and get information about the current implementation.
"""

import sys
import os


def is_cpp_available():
    """
    Check if the C++ backend is available.
    
    Returns:
        bool: True if C++ backend is loaded, False if using Python fallback
    """
    try:
        from . import lang_opt_native
        return True
    except ImportError:
        return False


def get_backend_info():
    """
    Get detailed information about the current backend.
    
    Returns:
        dict: Information about backend components and performance
    """
    info = {
        'cpp_available': is_cpp_available(),
        'components': {},
        'version': None,
        'location': None
    }
    
    if info['cpp_available']:
        try:
            from . import lang_opt_native
            info['version'] = getattr(lang_opt_native, '__version__', 'unknown')
            info['location'] = getattr(lang_opt_native, '__file__', 'unknown')
            info['components'] = {
                'lexer': 'C++ (~2.4x faster)',
                'parser': 'C++ (~6x faster)',
                'evaluator': 'Python (optimized built-ins)',
                'ast': 'Python (dataclasses)'
            }
        except Exception as e:
            info['error'] = str(e)
    else:
        info['components'] = {
            'lexer': 'Python',
            'parser': 'Python',
            'evaluator': 'Python',
            'ast': 'Python'
        }
    
    return info


def print_backend_status(verbose=False):
    """
    Print a human-readable status of the current backend.
    
    Args:
        verbose (bool): If True, print detailed information
    """
    info = get_backend_info()
    
    print("=" * 70)
    print("lang_opt Backend Status")
    print("=" * 70)
    
    if info['cpp_available']:
        print("✅ Status: C++ backend ACTIVE")
        print(f"📦 Version: {info['version']}")
        if verbose and info['location']:
            print(f"📁 Location: {info['location']}")
        print("\n🏗️  Components:")
        for component, impl in info['components'].items():
            print(f"   • {component.capitalize()}: {impl}")
        print("\n🚀 Fast parsing with C++, optimized evaluation with Python!")
    else:
        print("⚠️  Status: Using Python fallback")
        print("💡 Build the C++ extension for better performance")
        print("\n🏗️  All components running in pure Python")
        if verbose:
            print("\nTo build the C++ extension:")
            print("  cd src/lang_opt")
            print("  python setup.py build_ext --inplace")
    
    print("=" * 70)


def verify_functionality():
    """
    Verify that the backend is working correctly.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from . import tokenize, parse, evaluate
        
        # Test lexer
        tokens = tokenize("(+ 1 2)")
        if len(tokens) != 6:  # LPAREN, SYMBOL, NUMBER, NUMBER, RPAREN, EOF
            return False
        
        # Test parser
        ast = parse("(+ 1 2)")
        if ast is None:
            return False
        
        # Test evaluator
        result = evaluate("(+ 1 2)")
        if result != 3:
            return False
        
        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    """Command-line interface for backend checking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check lang_opt backend status"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify backend functionality"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    if args.json:
        import json
        info = get_backend_info()
        print(json.dumps(info, indent=2))
    else:
        print_backend_status(verbose=args.verbose)
        
        if args.verify:
            print("\n🧪 Running functionality tests...")
            if verify_functionality():
                print("✅ All tests passed!")
            else:
                print("❌ Tests failed!")
                sys.exit(1)

