from pathlib import Path

from utils import prepare_student_repo as prep


def test_beginner_single_line_homework_redaction() -> None:
    lines = [
        "# HOMEWORK: fill in the answer\n",
        "value = 42\n",
    ]

    processed = prep.process_lines_beginner(lines)

    assert processed[0] == "# HOMEWORK: fill in the answer\n"
    assert processed[1] == "value = None  # TODO: Implement this assignment\n"


def test_beginner_block_homework_starts_redaction() -> None:
    lines = [
        "def solve():\n",
        "    # HOMEWORK STARTS:\n",
        "    result = 1\n",
        "    return result\n",
        "    # HOMEWORK ENDS\n",
    ]

    processed = prep.process_lines_beginner(lines)
    processed_text = "".join(processed)

    assert "# HOMEWORK STARTS:" in processed_text
    assert "pass  # TODO: Implement this section" in processed_text
    assert "result = 1" not in processed_text
    assert "# HOMEWORK ENDS" in processed_text


def test_advanced_redacts_marked_function() -> None:
    lines = [
        "def update_value(x):\n",
        "    \"\"\"Update a value.\"\"\"\n",
        "    # HOMEWORK: implement this\n",
        "    return x + 1\n",
        "\n",
        "def untouched():\n",
        "    return 5\n",
    ]

    processed = prep.process_lines_advanced(lines)
    processed_text = "".join(processed)

    assert "def update_value(x):" in processed_text
    assert "pass  # TODO: Implement this function" in processed_text
    assert "return x + 1" not in processed_text
    assert "def untouched():" in processed_text


def test_copy_root_files_and_checks_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "reference"
    output_dir = tmp_path / "student"
    checks_dir = input_dir / "checks"
    checks_dir.mkdir(parents=True)

    for file_name in prep.ROOT_FILES:
        (input_dir / file_name).write_text(f"{file_name}\n", encoding="utf-8")

    (input_dir / "AGENTS.md").write_text("maintainer only\n", encoding="utf-8")
    (checks_dir / "bandits.txt").write_text("copied\n", encoding="utf-8")

    prep.copy_root_files(str(input_dir), str(output_dir))
    prep.process_directory(str(input_dir), str(output_dir), ["checks"], mode="beginner")

    for file_name in prep.ROOT_FILES:
        assert (output_dir / file_name).is_file()

    assert not (output_dir / "AGENTS.md").exists()
    assert (output_dir / "checks" / "bandits.txt").read_text(encoding="utf-8") == "copied\n"
