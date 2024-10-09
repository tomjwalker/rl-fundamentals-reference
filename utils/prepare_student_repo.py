#!/usr/bin/env python3
"""
prepare_student_repo.py

A script to prepare a student repository by processing reference reinforcement learning
scripts. It redacts specific sections marked for homework assignments, retains necessary
documentation, and copies essential files to the student repository.

**Redaction Logic**:

1. **HOMEWORK Markers**:
   - **Single-line Redaction**:
     - Lines starting with `# HOMEWORK:` followed by comments.
     - The following code line is redacted:
       - If it's an assignment (`LHS = RHS` or `LHS += RHS`), the RHS is replaced with a placeholder.
       - If it's a method call or other statement, it's replaced with a `# TODO` comment.
   - **Block Redaction**:
     - Blocks between `# HOMEWORK START` and `# HOMEWORK END` are replaced with a placeholder line.

2. **ASSIGNMENT and SOLUTION Markers**:
   - **ASSIGNMENT Blocks**:
     - Code between `# ASSIGNMENT START` and `# ASSIGNMENT END` is retained.
   - **SOLUTION Blocks**:
     - Entirely removed from the student version, including the marker comments.

3. **Augmented Assignments**:
   - Handles operators like `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `|=`, `^=`, `>>=`, `<<=`, `@=`.
   - Redacts the RHS while keeping the operator intact.

4. **Other Statements**:
   - Non-assignment lines that are marked for redaction are replaced with a `# TODO` comment.

**Usage**:
    python utils/prepare_student_repo.py <input_dir> <output_dir> --dirs <dir1> <dir2> ...

**Example**:
    python utils/prepare_student_repo.py . ../rl-fundamentals-assignments --dirs rl exercises

**Arguments**:
    input_dir     Path to the input directory (reference repository).
    output_dir    Path to the output directory (student repository).
    --dirs        One or more directories within the input directory to process (e.g., rl exercises).

**Notes**:
    - The script processes Python files (.py) by redacting code sections based on the markers.
    - It copies README.md and .gitignore files without modification.
    - A requirements.txt file is generated based on the input directory's dependencies.
    - A main README.md is created in the output directory with introductory content.

This script courtesy of o1-preview ;) (cheers mate!)
"""

import os
import shutil
import argparse
import re
import sys


def process_code_line(line: str) -> str:
    """
    Redact a single line of code by replacing assignments with placeholders.

    Args:
        line (str): The original line of code.

    Returns:
        str: The redacted line with placeholders for student implementation.
    """
    stripped_line = line.rstrip('\n')
    # Adjusted regex to handle optional type annotations and augmented assignments
    assignment_match = re.match(
        r'^(\s*)([\w,\s]+)(\s*:\s*[\w\[\],\s]+)?\s*([+\-*/%@&|^]=|//=|>>=|<<=|@=|=)\s*(.*)',
        stripped_line
    )
    if assignment_match:
        indent = assignment_match.group(1)
        lhs = assignment_match.group(2)
        type_annotation = assignment_match.group(3) or ''
        operator = assignment_match.group(4)
        # Retain indentation, LHS, operator, and type annotation
        return f"{indent}{lhs}{type_annotation} {operator} None  # TODO: Implement this assignment\n"
    else:
        # Handle non-assignment lines
        indent_match = re.match(r'^(\s*)', stripped_line)
        indent = indent_match.group(1) if indent_match else ''
        return f"{indent}# TODO: Implement this line\n"


def process_file(input_path: str, output_path: str) -> None:
    """
    Process a single Python file by redacting homework and solution sections and copying necessary content.

    Args:
        input_path (str): The path to the original Python file.
        output_path (str): The path to save the redacted Python file.
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    processed_lines = []
    i = 0
    skip_block = False
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        # Skip personal TODO comments
        if stripped_line.startswith("# TODO") and not any(
            tag in stripped_line for tag in ["# TODO: Implement", "# TODO: Implement this assignment"]
        ):
            i += 1
            continue

        # Handle SOLUTION blocks
        elif any(stripped_line.startswith(tag) for tag in ["# SOLUTION START", "# SOLUTION BEGINS"]):
            # Skip all lines until SOLUTION END
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("# SOLUTION END"):
                i += 1
            # Skip the SOLUTION END line
            if i < len(lines):
                i += 1
            continue

        # Handle ASSIGNMENT blocks
        elif any(stripped_line.startswith(tag) for tag in ["# ASSIGNMENT START", "# ASSIGNMENT BEGINS"]):
            # Retain the ASSIGNMENT START line
            processed_lines.append(line)
            i += 1
            # Retain all lines until ASSIGNMENT END
            while i < len(lines) and not lines[i].strip().startswith("# ASSIGNMENT END"):
                processed_lines.append(lines[i])
                i += 1
            # Retain the ASSIGNMENT END line
            if i < len(lines):
                processed_lines.append(lines[i])
                i += 1
            continue

        # Handle HOMEWORK markers
        elif any(stripped_line.startswith(tag) for tag in ["# HOMEWORK:", "# HOMEWORK BEGINS:", "# HOMEWORK START:"]):
            # Retain all consecutive comment lines related to homework
            while i < len(lines) and lines[i].strip().startswith("#"):
                processed_lines.append(lines[i])
                i += 1

            # Handle single-line homework redaction
            if "HOMEWORK:" in stripped_line:
                if i < len(lines):
                    code_line = lines[i]
                    redacted_line = process_code_line(code_line)
                    processed_lines.append(redacted_line)
                    i += 1
            else:
                # Handle multi-line homework block redaction
                next_line = lines[i] if i < len(lines) else ''
                indent_match = re.match(r'^(\s*)', next_line)
                indent = indent_match.group(1) if indent_match else ''
                # Add a placeholder for multi-line homework sections
                processed_lines.append(f"{indent}pass  # TODO: Implement this section\n")
                # Skip lines until the end of the homework block
                while i < len(lines) and not any(
                    lines[i].strip().startswith(tag) for tag in ["# HOMEWORK ENDS", "# HOMEWORK END"]
                ):
                    i += 1
                # Retain the end marker of the homework block
                if i < len(lines):
                    processed_lines.append(lines[i])
                    i += 1
            continue

        else:
            # Regular line, add without modification
            processed_lines.append(line)
            i += 1

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(processed_lines)


def process_directory(input_dir: str, output_dir: str, dirs_to_process: list) -> None:
    """
    Traverse the input directory, process specified subdirectories, and copy/redact files accordingly.

    Args:
        input_dir (str): The path to the input directory (reference repository).
        output_dir (str): The path to the output directory (student repository).
        dirs_to_process (list): List of subdirectories to process within the input directory.
    """
    for root, dirs, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)

        # Skip directories not in dirs_to_process
        if dirs_to_process and not any(rel_dir == d or rel_dir.startswith(d + os.sep) for d in dirs_to_process):
            continue

        for file in files:
            if file.endswith('.py'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                process_file(input_path, output_path)
            elif file in ['README.md', '.gitignore']:
                # Copy these files without processing
                source_file = os.path.join(root, file)
                destination_dir = os.path.join(output_dir, rel_dir)
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy2(source_file, os.path.join(destination_dir, file))


def create_requirements_txt(input_dir: str, output_dir: str) -> None:
    """
    Generate a requirements.txt file for the student repository based on the input directory's dependencies.

    Args:
        input_dir (str): The path to the input directory (reference repository).
        output_dir (str): The path to the output directory (student repository).
    """
    try:
        import pipreqs  # Ensure pipreqs is installed
    except ImportError:
        print("pipreqs is not installed. Please install it using 'pip install pipreqs'")
        sys.exit(1)

    # Generate requirements.txt using pipreqs
    os.system(f"pipreqs {input_dir} --force --savepath {os.path.join(output_dir, 'requirements.txt')}")


def create_main_readme(output_dir: str) -> None:
    """
    Create a main README.md file in the output directory with introductory content.

    Args:
        output_dir (str): The path to the output directory (student repository).
    """
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# RL Course Exercises\n\n")
        f.write(
            "Welcome to the RL course exercises. Please refer to individual exercise directories for specific "
            "instructions.\n"
        )


def main() -> None:
    """
    The main function to parse command-line arguments and initiate the processing of the repository.
    """
    parser = argparse.ArgumentParser(
        description="Process RL course files for student distribution by redacting homework and solution sections."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input directory (reference repository).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory (student repository).",
    )
    parser.add_argument(
        "--dirs",
        nargs='+',
        required=True,
        help="One or more directories within the input directory to process (e.g., rl exercises).",
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process the specified directories
    print("Processing specified directories...")
    process_directory(args.input_dir, args.output_dir, args.dirs)

    # Create requirements.txt
    print("Generating requirements.txt...")
    create_requirements_txt(args.input_dir, args.output_dir)

    # Create a main README.md in the output directory
    print("Creating main README.md...")
    create_main_readme(args.output_dir)

    print("Processing complete. Student repository created successfully.")


if __name__ == "__main__":
    main()
