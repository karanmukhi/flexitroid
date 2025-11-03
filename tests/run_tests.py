#!/usr/bin/env python3
import sys
import pytest


def main():
    """Run all tests in the project."""
    # Run pytest with automatic test discovery
    # -v for verbose output
    # --tb=short for shorter traceback
    # exit with the appropriate exit code
    exit_code = pytest.main(["-v", "--tb=short"])
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
