#!/usr/bin/env python3
"""
prepare_student_repo.py

A script to prepare student versions of a reinforcement learning (RL) course repository
by processing reference code. It supports generating multiple levels of assignments
(e.g., beginner and advanced) by redacting specific code sections based on markers
and function content.

**Functionality**:

- **Modes**:
  - **Beginner Mode**: Redacts code sections marked with specific markers, replacing them with placeholders or TODO comments.
  - **Advanced Mode**: Redacts entire functions or methods if they contain any of the specified markers.

- **Markers and Redaction Logic**:

  1. **HOMEWORK Markers**:
     - **Single-line Redaction**:
       - Lines starting with `# HOMEWORK:` followed by comments.
       - The following code line is redacted:
         - If it's an assignment (`LHS = RHS` or `LHS += RHS`), the RHS is replaced with a placeholder.
         - If it's a method call or other statement, it's replaced with a `# TODO` comment.
     - **Block Redaction**:
       - Blocks between `# HOMEWORK START`, `# HOMEWORK STARTS`, and `# HOMEWORK END(S)` are replaced with a placeholder line.

  2. **ASSIGNMENT and SOLUTION Markers**:
     - **ASSIGNMENT Blocks**:
       - Code between `# ASSIGNMENT START` and `# ASSIGNMENT END` is retained.
     - **SOLUTION Blocks**:
       - Entirely removed from the student version, including the marker comments.

  3. **Advanced Mode Redaction**:
     - In advanced mode, if any function or method contains any of the markers (`# HOMEWORK`, `# ASSIGNMENT`, `# SOLUTION`), the entire function or method is redacted.
     - The function or method is replaced with a placeholder function definition with a `pass` statement and a TODO comment.

  4. **Augmented Assignments**:
     - Handles operators like `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `|=`, `^=`, `>>=`, `<<=`, `@=`.
     - Redacts the RHS while keeping the operator intact.

  5. **Other Statements**:
     - Non-assignment lines that are marked for redaction are replaced with a `# TODO` comment.

**Usage**:
    python utils/prepare_student_repo.py <input_dir> <output_dir> [--dirs <dir1> <dir2> ...] [--mode beginner|advanced]

**Example**:
    python utils/prepare_student_repo.py . ../rl-fundamentals-assignments-beginner
    python utils/prepare_student_repo.py . ../rl-fundamentals-assignments-advanced --mode advanced

**Arguments**:
    input_dir     Path to the input directory (reference repository).
    output_dir    Path to the output directory (student repository).
    --dirs        (Optional) One or more directories within the input directory to process. Defaults to
                  ['rl', 'exercises', 'assignments', 'images', 'checks'].
    --mode        (Optional) Processing mode: 'beginner' or 'advanced'. Defaults to 'beginner'.

**Notes**:
    - The script processes Python files (.py) by redacting code sections based on the markers.
    - It copies other files (e.g., .md, .png, .jpg) in the specified directories without modification.
    - It copies the root README.md, .gitignore, pyproject.toml, uv.lock, and requirements.txt files without modification.

Generated with the help of o1-preview!
"""

import argparse
import os
import re
import shutil
import sys


ROOT_FILES = ["README.md", ".gitignore", ".python-version", "pyproject.toml", "uv.lock", "requirements.txt"]
HOMEWORK_MARKERS = ["# HOMEWORK:", "# HOMEWORK BEGINS:", "# HOMEWORK START:", "# HOMEWORK STARTS:"]
HOMEWORK_END_MARKERS = ["# HOMEWORK ENDS", "# HOMEWORK END"]
DEFAULT_DIRS = ["rl", "exercises", "assignments", "images", "checks"]


def process_code_line(line: str) -> str:
    """
    Redact a single line of code by replacing assignments with placeholders.

    Args:
        line (str): The original line of code.

    Returns:
        str: The redacted line with placeholders for student implementation.
    """
    stripped_line = line.rstrip("\n")
    assignment_match = re.match(
        r"^(\s*)([\w,\s]+)(\s*:\s*[\w\[\],\s]+)?\s*([+\-*/%@&|^]=|//=|>>=|<<=|@=|=)\s*(.*)",
        stripped_line,
    )
    if assignment_match:
        indent = assignment_match.group(1)
        lhs = assignment_match.group(2).rstrip()
        type_annotation = assignment_match.group(3) or ""
        operator = assignment_match.group(4)
        return f"{indent}{lhs}{type_annotation} {operator} None  # TODO: Implement this assignment\n"

    indent_match = re.match(r"^(\s*)", stripped_line)
    indent = indent_match.group(1) if indent_match else ""
    return f"{indent}# TODO: Implement this line\n"


def process_file(input_path: str, output_path: str, mode: str = "beginner") -> None:
    """
    Process a single Python file by redacting homework and solution sections and copying necessary content.

    Args:
        input_path (str): The path to the original Python file.
        output_path (str): The path to save the redacted Python file.
        mode (str): Processing mode, 'beginner' or 'advanced'.
    """
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    if mode == "beginner":
        processed_lines = process_lines_beginner(lines)
    elif mode == "advanced":
        processed_lines = process_lines_advanced(lines)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(processed_lines)


def process_lines_beginner(lines):
    """
    Process lines for beginner mode.

    Args:
        lines (list): List of lines from the input file.

    Returns:
        list: Processed lines.
    """
    processed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        if stripped_line.startswith("# TODO") and not any(
            tag in stripped_line for tag in ["# TODO: Implement", "# TODO: Implement this assignment"]
        ):
            i += 1

        elif any(stripped_line.startswith(tag) for tag in ["# SOLUTION START", "# SOLUTION BEGINS"]):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("# SOLUTION END"):
                i += 1
            if i < len(lines):
                i += 1

        elif any(stripped_line.startswith(tag) for tag in ["# ASSIGNMENT START", "# ASSIGNMENT BEGINS"]):
            processed_lines.append(line)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("# ASSIGNMENT END"):
                processed_lines.append(lines[i])
                i += 1
            if i < len(lines):
                processed_lines.append(lines[i])
                i += 1

        elif any(stripped_line.startswith(tag) for tag in HOMEWORK_MARKERS):
            while i < len(lines) and lines[i].strip().startswith("#"):
                processed_lines.append(lines[i])
                i += 1

            if "HOMEWORK:" in stripped_line:
                if i < len(lines):
                    processed_lines.append(process_code_line(lines[i]))
                    i += 1
            else:
                next_line = lines[i] if i < len(lines) else ""
                indent_match = re.match(r"^(\s*)", next_line)
                indent = indent_match.group(1) if indent_match else ""
                processed_lines.append(f"{indent}pass  # TODO: Implement this section\n")
                while i < len(lines) and not any(lines[i].strip().startswith(tag) for tag in HOMEWORK_END_MARKERS):
                    i += 1
                if i < len(lines):
                    processed_lines.append(lines[i])
                    i += 1

        else:
            processed_lines.append(line)
            i += 1

    return processed_lines


def process_lines_advanced(lines):
    """
    Process lines for advanced mode.

    Args:
        lines (list): List of lines from the input file.

    Returns:
        list: Processed lines.
    """
    function_ranges = []
    i = 0
    while i < len(lines):
        line = lines[i]
        func_match = re.match(r"^(\s*)def\s+\w+\s*\(.*\)\s*(->\s*[\w\[\],\s\.]+)?\s*:", line)
        if func_match:
            indent_level = len(func_match.group(1))
            start_line = i
            should_redact = False
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_line_strip = next_line.strip()
                next_indent_level = len(next_line) - len(next_line.lstrip(" "))
                if next_indent_level <= indent_level and next_line_strip != "":
                    break
                if any(marker in next_line_strip for marker in ["# HOMEWORK", "# ASSIGNMENT", "# SOLUTION"]):
                    should_redact = True
                i += 1
            function_ranges.append((start_line, i, should_redact))
        else:
            i += 1

    processed_lines = []
    function_index = 0
    i = 0
    while i < len(lines):
        if function_index < len(function_ranges) and i == function_ranges[function_index][0]:
            start_line, end_line, should_redact = function_ranges[function_index]
            function_index += 1
            if should_redact:
                func_def_line = lines[start_line]
                func_match = re.match(r"^(\s*)def\s+(\w+)\s*\(.*\)\s*(->\s*[\w\[\],\s\.]+)?\s*:", func_def_line)
                indent = func_match.group(1)
                func_name = func_match.group(2)
                params = func_def_line[func_def_line.find("(") : func_def_line.rfind(")") + 1]
                return_annotation = func_match.group(3) if func_match.group(3) else ""
                func_body_lines = lines[start_line + 1 : end_line]
                docstring_lines = []
                if func_body_lines:
                    first_body_line = func_body_lines[0].strip()
                    if first_body_line.startswith(('"""', "'''")):
                        docstring_lines.append(func_body_lines[0])
                        if not (first_body_line.endswith(('"""', "'''")) and len(first_body_line) > 3):
                            j = 1
                            while j < len(func_body_lines):
                                docstring_lines.append(func_body_lines[j])
                                stripped = func_body_lines[j].strip()
                                if stripped.endswith(('"""', "'''")) and len(stripped) > 3:
                                    break
                                if stripped in ('"""', "'''"):
                                    break
                                j += 1
                placeholder = f"{indent}def {func_name}{params}{return_annotation}:\n"
                for doc_line in docstring_lines:
                    placeholder += doc_line
                placeholder += f"{indent}    pass  # TODO: Implement this function\n"
                processed_lines.append(placeholder)
                processed_lines.append("\n\n")
                i = end_line
            else:
                while i < end_line:
                    processed_lines.append(lines[i])
                    i += 1
        else:
            processed_lines.append(lines[i])
            i += 1

    return processed_lines


def process_directory(input_dir: str, output_dir: str, dirs_to_process: list, mode: str = "beginner") -> None:
    """
    Traverse the input directory, process specified subdirectories, and copy/redact files accordingly.

    Args:
        input_dir (str): The path to the input directory (reference repository).
        output_dir (str): The path to the output directory (student repository).
        dirs_to_process (list): List of subdirectories to process within the input directory.
        mode (str): Processing mode, 'beginner' or 'advanced'.
    """
    for root, dirs, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)
        if dirs_to_process and not any(rel_dir == d or rel_dir.startswith(d + os.sep) for d in dirs_to_process):
            continue

        for file_name in files:
            input_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if file_name.endswith(".py"):
                process_file(input_path, output_path, mode)
            else:
                shutil.copy2(input_path, output_path)


def copy_root_files(input_dir: str, output_dir: str) -> None:
    """
    Copy root-level project files to the output directory.

    Args:
        input_dir (str): The path to the input directory.
        output_dir (str): The path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_name in ROOT_FILES:
        source_file = os.path.join(input_dir, file_name)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, os.path.join(output_dir, file_name))


def main() -> None:
    """
    The main function to parse command-line arguments and initiate the processing of the repository.
    """
    parser = argparse.ArgumentParser(
        description="Process RL course files for student distribution by redacting homework and solution sections."
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory (reference repository).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory (student repository).")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_DIRS,
        help="One or more directories within the input directory to process. Defaults to ['rl', 'exercises', 'assignments', 'images', 'checks'].",
    )
    parser.add_argument(
        "--mode",
        choices=["beginner", "advanced"],
        default="beginner",
        help="Processing mode: 'beginner' or 'advanced'. Defaults to 'beginner'.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Copying root-level files...")
    copy_root_files(args.input_dir, args.output_dir)

    print(f"Processing specified directories in {args.mode} mode...")
    process_directory(args.input_dir, args.output_dir, args.dirs, mode=args.mode)

    print("Processing complete. Student repository created successfully.")


if __name__ == "__main__":
    main()

