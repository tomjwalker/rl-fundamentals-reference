import os
import shutil
import argparse
import re


def process_code_line(line):
    stripped_line = line.rstrip('\n')
    # Adjusted regex to handle optional type annotations
    assignment_match = re.match(r'^(\s*)([\w,\s]+)(\s*:\s*[\w\[\],\s]+)?\s*=\s*(.*)', stripped_line)
    if assignment_match:
        indent = assignment_match.group(1)
        lhs = assignment_match.group(2)
        type_annotation = assignment_match.group(3) or ''
        # Retain indentation, LHS, and type annotation
        return f"{indent}{lhs}{type_annotation} = None  # TODO: Implement this assignment\n"
    else:
        # Handle non-assignment lines
        indent = re.match(r'^(\s*)', stripped_line).group(1)
        return f"{indent}# TODO: Implement this line\n"


def process_file(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    processed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        if stripped_line.startswith("# TODO"):
            i += 1
            continue  # Skip personal TODO comments

        elif any(stripped_line.startswith(tag) for tag in ["# HOMEWORK:", "# HOMEWORK BEGINS:", "# HOMEWORK START:"]):
            # Retain all consecutive comment lines
            while i < len(lines) and lines[i].strip().startswith("#"):
                processed_lines.append(lines[i])
                i += 1
            # Handle code redaction
            if stripped_line.startswith("# HOMEWORK:"):
                # Handle single-line homework
                if i < len(lines):
                    code_line = lines[i]
                    redacted_line = process_code_line(code_line)
                    processed_lines.append(redacted_line)
                    i += 1
            else:
                # Handle multi-line homework block
                # Determine indentation level
                next_line = lines[i] if i < len(lines) else ''
                indent_match = re.match(r'^(\s*)', next_line)
                indent = indent_match.group(1) if indent_match else ''
                # Add placeholder
                processed_lines.append(f"{indent}pass  # TODO: Implement this section\n")
                # Skip lines until end of homework block
                while i < len(lines) and not any(lines[i].strip().startswith(tag) for tag in ["# HOMEWORK ENDS", "# HOMEWORK END"]):
                    i += 1
                # Retain the end marker
                if i < len(lines):
                    processed_lines.append(lines[i])
                    i += 1
        else:
            # Regular line, just add it
            processed_lines.append(line)
            i += 1

    with open(output_path, 'w') as outfile:
        outfile.writelines(processed_lines)


def process_directory(input_dir, output_dir, dirs_to_process):
    for root, dirs, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)

        # Skip directories not in dirs_to_process
        if dirs_to_process and not any(rel_dir.startswith(d) for d in dirs_to_process):
            continue

        for file in files:
            if file.endswith('.py'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                process_file(input_path, output_path)
            elif file in ['README.md', '.gitignore']:
                # Copy these files without processing
                shutil.copy2(os.path.join(root, file), os.path.join(output_dir, rel_dir, file))


def create_requirements_txt(input_dir, output_dir):
    # This is a basic approach. You might need to adjust based on your project's specifics.
    os.system(f"pipreqs {input_dir} --savepath {os.path.join(output_dir, 'requirements.txt')}")


def main():
    parser = argparse.ArgumentParser(description="Process RL course files for student distribution.")
    parser.add_argument("input_dir", help="Path to the input directory (reference repo)")
    parser.add_argument("output_dir", help="Path to the output directory (student repo)")
    parser.add_argument("--dirs", nargs='+', help="Directories to process (e.g., rl exercises)")
    args = parser.parse_args()

    # Process the specified directories
    process_directory(args.input_dir, args.output_dir, args.dirs)

    # Create requirements.txt
    create_requirements_txt(args.input_dir, args.output_dir)

    # Create a main README.md in the output directory
    with open(os.path.join(args.output_dir, 'README.md'), 'w') as f:
        f.write("# RL Course Exercises\n\n")
        f.write(
            "Welcome to the RL course exercises. Please refer to individual exercise directories for specific instructions.\n")

    print("Processing complete. Student repo created successfully.")


if __name__ == "__main__":
    main()
