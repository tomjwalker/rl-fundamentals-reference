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
       - Blocks between `# HOMEWORK START` and `# HOMEWORK END` are replaced with a placeholder line.

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
    --dirs        (Optional) One or more directories within the input directory to process. Defaults to ['rl', 'exercises', 'assignments', 'images'].
    --mode        (Optional) Processing mode: 'beginner' or 'advanced'. Defaults to 'beginner'.

**Notes**:
    - The script processes Python files (.py) by redacting code sections based on the markers.
    - It copies other files (e.g., .md, .png, .jpg) in the specified directories without modification.
    - It copies the root README.md and .gitignore files without modification.
    - A requirements.txt file is generated based on the input directory's dependencies.

Generated with the help of o1-preview!
"""

import os
import shutil
import argparse
import re
import sys
import subprocess

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

def process_file(input_path: str, output_path: str, mode: str = 'beginner') -> None:
    """
    Process a single Python file by redacting homework and solution sections and copying necessary content.

    Args:
        input_path (str): The path to the original Python file.
        output_path (str): The path to save the redacted Python file.
        mode (str): Processing mode, 'beginner' or 'advanced'.
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    if mode == 'beginner':
        processed_lines = process_lines_beginner(lines)
    elif mode == 'advanced':
        processed_lines = process_lines_advanced(lines)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as outfile:
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

    return processed_lines

def process_lines_advanced(lines):
    """
    Process lines for advanced mode.

    Args:
        lines (list): List of lines from the input file.

    Returns:
        list: Processed lines.
    """
    # First pass: Identify functions/methods with markers
    function_ranges = []  # List of tuples (start_line, end_line, should_redact)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        # Detect function or method definition, accounting for return type annotations
        func_match = re.match(r'^(\s*)def\s+\w+\s*\(.*\)\s*(->\s*[\w\[\],\s\.]+)?\s*:', line)
        if func_match:
            indent_level = len(func_match.group(1))
            start_line = i
            should_redact = False
            i += 1
            # Collect function body
            while i < len(lines):
                next_line = lines[i]
                next_line_strip = next_line.strip()
                next_indent_level = len(next_line) - len(next_line.lstrip(' '))

                # Check if we've exited the function (dedentation)
                if next_indent_level <= indent_level and next_line_strip != '':
                    break

                # Check for markers within the function
                if any(marker in next_line_strip for marker in ['# HOMEWORK', '# ASSIGNMENT', '# SOLUTION']):
                    should_redact = True

                i += 1

            end_line = i
            function_ranges.append((start_line, end_line, should_redact))
        else:
            i += 1

    # Second pass: Write the output file with redactions
    processed_lines = []
    function_index = 0
    i = 0
    while i < len(lines):
        # Check if current line is the start of a function to redact
        if function_index < len(function_ranges) and i == function_ranges[function_index][0]:
            start_line, end_line, should_redact = function_ranges[function_index]
            function_index += 1
            if should_redact:
                # Redact the entire function but retain the docstring
                func_def_line = lines[start_line]
                func_match = re.match(r'^(\s*)def\s+(\w+)\s*\(.*\)\s*(->\s*[\w\[\],\s\.]+)?\s*:', func_def_line)
                indent = func_match.group(1)
                func_name = func_match.group(2)
                params = func_def_line[func_def_line.find('('):func_def_line.rfind(')')+1]
                return_annotation = func_match.group(3) if func_match.group(3) else ''
                # Extract the docstring if present
                func_body_lines = lines[start_line+1:end_line]
                docstring_lines = []
                if func_body_lines:
                    first_body_line = func_body_lines[0]
                    first_body_line_stripped = first_body_line.strip()
                    if first_body_line_stripped.startswith(('"""', "'''")):
                        # Start of docstring
                        docstring_lines.append(first_body_line)
                        j = 1
                        while j < len(func_body_lines):
                            line = func_body_lines[j]
                            docstring_lines.append(line)
                            if line.strip().endswith(('"""', "'''")) and len(line.strip()) > 3:
                                break
                            elif line.strip() in ('"""', "'''"):
                                break
                            j += 1
                # Build the placeholder function
                placeholder = f"{indent}def {func_name}{params}{return_annotation}:\n"
                if docstring_lines:
                    # Add docstring lines as is
                    for doc_line in docstring_lines:
                        placeholder += doc_line
                placeholder += f"{indent}    pass  # TODO: Implement this function\n"
                # Add the placeholder to processed_lines
                processed_lines.append(placeholder)
                # Add two blank lines after the function
                processed_lines.append('\n\n')
                i = end_line  # Skip to the end of the function
            else:
                # Keep the function as is
                while i < end_line:
                    processed_lines.append(lines[i])
                    i += 1
        else:
            # Regular line, add without modification
            processed_lines.append(lines[i])
            i += 1

    return processed_lines


def process_directory(input_dir: str, output_dir: str, dirs_to_process: list, mode: str = 'beginner') -> None:
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

        # Skip directories not in dirs_to_process
        if dirs_to_process and not any(rel_dir == d or rel_dir.startswith(d + os.sep) for d in dirs_to_process):
            continue

        for file in files:
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            destination_dir = os.path.dirname(output_path)
            os.makedirs(destination_dir, exist_ok=True)

            if file.endswith('.py'):
                # Process Python files
                process_file(input_path, output_path, mode)
            else:
                # Copy other files without processing
                shutil.copy2(input_path, output_path)

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
        print("pipreqs is not installed in the current environment. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pipreqs"])
        import pipreqs

    # Find the pipreqs executable
    pipreqs_executable = shutil.which('pipreqs')
    if not pipreqs_executable:
        # Try to find pipreqs in the virtual environment's Scripts directory
        venv_scripts_dir = os.path.dirname(sys.executable)
        pipreqs_executable = os.path.join(venv_scripts_dir, 'pipreqs')
        if os.name == 'nt':
            pipreqs_executable += '.exe'

        if not os.path.isfile(pipreqs_executable):
            print("Error: pipreqs executable not found.")
            sys.exit(1)

    # Generate requirements.txt using pipreqs
    output_file = os.path.join(output_dir, 'requirements.txt')
    command = [
        pipreqs_executable, input_dir,
        '--force', '--savepath', output_file
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error generating requirements.txt:")
        print(e.stderr)
        sys.exit(1)



def copy_root_files(input_dir: str, output_dir: str) -> None:
    """
    Copy root-level files like README.md and .gitignore to the output directory.

    Args:
        input_dir (str): The path to the input directory.
        output_dir (str): The path to the output directory.
    """
    for file_name in ['README.md', '.gitignore']:
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
        default=['rl', 'exercises', 'assignments', 'images'],
        help="One or more directories within the input directory to process. Defaults to ['rl', 'exercises', 'assignments', 'images'].",
    )
    parser.add_argument(
        "--mode",
        choices=['beginner', 'advanced'],
        default='beginner',
        help="Processing mode: 'beginner' or 'advanced'. Defaults to 'beginner'.",
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy root-level files like README.md and .gitignore
    print("Copying root-level files...")
    copy_root_files(args.input_dir, args.output_dir)

    # Process the specified directories
    print(f"Processing specified directories in {args.mode} mode...")
    process_directory(args.input_dir, args.output_dir, args.dirs, mode=args.mode)

    # Create requirements.txt
    print("Generating requirements.txt...")
    create_requirements_txt(args.input_dir, args.output_dir)

    print("Processing complete. Student repository created successfully.")

if __name__ == "__main__":
    main()
