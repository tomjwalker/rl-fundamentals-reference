"""
Generate a file tree representation of a directory structure.

Usage:
    python utils/generate_file_tree.py [OPTIONS]

Options:
    --path PATH                 The path to generate the tree from (default: current directory)
    --output, -o FILE           Output file to save the tree (default: file_tree.txt)
    --print                     Print the tree to console instead of saving to a file
    --max-depth, -d DEPTH       Maximum depth of the directory tree
    --exclude-dirs, -e DIRS     Additional directories to exclude (space-separated)
    --exclude-extensions, -x EXTS   Additional file extensions to exclude (space-separated)

Example:
    python utils/generate_file_tree.py --max-depth 3 -o custom_tree.txt

Note: By default, this script excludes common directories (.git, __pycache__, .data, venv)
and file extensions (.pyc, .pyd). You can add more exclusions using the options above.
"""

import os
import argparse
from pathlib import Path

# Default exclusions
DEFAULT_EXCLUDE_DIRS = {'.git', '__pycache__', '.data', 'venv', 'videos', 'stitched_videos'}
DEFAULT_EXCLUDE_EXTENSIONS = {'.pyc', '.pyd'}


def should_exclude(path, exclude_dirs, exclude_extensions):
    return (any(excluded in path.parts for excluded in exclude_dirs) or
            path.suffix.lower() in exclude_extensions)


def generate_tree(startpath, max_depth=None, exclude_dirs=None, exclude_extensions=None):
    if exclude_dirs is None:
        exclude_dirs = set()
    if exclude_extensions is None:
        exclude_extensions = set()

    exclude_dirs = DEFAULT_EXCLUDE_DIRS.union(exclude_dirs)
    exclude_extensions = DEFAULT_EXCLUDE_EXTENSIONS.union(exclude_extensions)

    startpath = Path(startpath)
    tree = []

    for root, dirs, files in os.walk(startpath):
        level = len(Path(root).relative_to(startpath).parts)
        if max_depth is not None and level > max_depth:
            continue

        path = Path(root)
        if should_exclude(path, exclude_dirs, exclude_extensions):
            continue

        indent = '  ' * level
        tree.append(f"{indent}{path.name}/")

        for file in sorted(files):
            file_path = path / file
            if not should_exclude(file_path, exclude_dirs, exclude_extensions):
                tree.append(f"{indent}  {file}")

    return '\n'.join(tree)


def main():
    parser = argparse.ArgumentParser(description="Generate a file tree representation.")
    parser.add_argument("--path", default=".", help="The path to generate the tree from.")
    parser.add_argument("--output", "-o", default="file_tree.txt", help="Output file to save the tree.")
    parser.add_argument("--print", action="store_true", help="Print the tree to console instead of saving to a file.")
    parser.add_argument("--max-depth", "-d", type=int, help="Maximum depth of the directory tree.")
    parser.add_argument("--exclude-dirs", "-e", nargs="*", default=[],
                        help="Additional directories to exclude from the tree.")
    parser.add_argument("--exclude-extensions", "-x", nargs="*", default=[],
                        help="Additional file extensions to exclude from the tree.")

    args = parser.parse_args()

    exclude_dirs = set(args.exclude_dirs)
    exclude_extensions = set(args.exclude_extensions)

    tree = generate_tree(args.path, args.max_depth, exclude_dirs, exclude_extensions)

    if args.print:
        print(tree)
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(tree)
        print(f"File tree saved to {args.output}")


if __name__ == "__main__":
    main()
