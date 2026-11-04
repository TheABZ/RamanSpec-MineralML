"""Convenience wrapper; install the package first, then run from the repository root."""
from ramanspec.__main__ import main

if __name__ == '__main__':
    import sys
    sys.argv.insert(1, 'demo')
    main()
