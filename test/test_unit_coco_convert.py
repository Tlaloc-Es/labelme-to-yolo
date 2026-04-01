import json
import os
import pathlib
import tempfile
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from labelme_to_yolo.__main__ import get_image_path, main, process


def test_conversion():
    TOLERANCE = 1e-6  # noqa N806
    current_path = pathlib.Path(__file__).parent.resolve()

    with open(
        os.path.join(current_path, "__data__/yolo_annotations/000000000285.txt"),
        encoding="utf-8",
    ) as file_handler:
        data = file_handler.readline()
        data = data.replace("\n", "")
        numbers = data.split(" ")[1:]
        numbers = np.array([*map(float, numbers)])

    with open(
        os.path.join(current_path, "__data__/yolo_annotations/000000000285.json"),
        encoding="utf-8",
    ) as file_handler:
        data = json.load(file_handler)
        height = int(data["imageHeight"])
        width = int(data["imageWidth"])
        values = np.array(data["values"])

    converted_values = process(values, width, height)

    for x0, x1 in zip(numbers, converted_values):
        assert x0 - x1 < TOLERANCE


def create_test_data(temp_dir: Path) -> tuple[Path, Path]:
    labelme_data = {
        "shapes": [
            {
                "label": "person",
                "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                "shape_type": "polygon",
            }
        ],
        "imageHeight": 100,
        "imageWidth": 100,
    }

    json_path = temp_dir / "test.json"
    with open(json_path, "w") as f:
        json.dump(labelme_data, f)

    image_path = temp_dir / "test.jpg"
    image_path.write_bytes(b"dummy image data")

    return json_path, image_path


def _assert_output(output_dir: Path, file_name: str = "test") -> None:
    """A single file may land in train, val or test due to integer rounding."""
    assert (output_dir / "project.yml").exists()

    splits = ["train", "val", "test"]
    image_found = any(
        (output_dir / "images" / s / f"{file_name}.jpg").exists() for s in splits
    )
    label_found = any(
        (output_dir / "labels" / s / f"{file_name}.txt").exists() for s in splits
    )
    assert image_found, "Converted image not found in any split directory"
    assert label_found, "Label file not found in any split directory"

    for split in splits:
        label_path = output_dir / "labels" / split / f"{file_name}.txt"
        if label_path.exists():
            content = label_path.read_text().strip()
            assert content.startswith("0 ")
            break


def test_cli_with_absolute_paths():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir).resolve()
        create_test_data(temp_path)
        output_dir = temp_path / "output"
        output_dir.mkdir()

        result = runner.invoke(
            main,
            ["--source-path", str(temp_path), "--output-path", str(output_dir)],
        )

        assert result.exit_code == 0, result.output
        _assert_output(output_dir)


def test_cli_with_relative_paths():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir).resolve()
        create_test_data(temp_path)
        output_dir = temp_path / "output"
        output_dir.mkdir()

        saved_cwd = os.getcwd()
        os.chdir(temp_path)
        try:
            result = runner.invoke(
                main,
                ["--source-path", ".", "--output-path", "output"],
            )
        finally:
            os.chdir(saved_cwd)

        assert result.exit_code == 0, result.output
        _assert_output(output_dir)


def test_get_image_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        (temp_path / "test.jpg").write_text("dummy")
        (temp_path / "test.jpeg").write_text("dummy")
        (temp_path / "test.png").write_text("dummy")

        result = get_image_path(str(temp_path), "test")
        assert result == str(temp_path / "test.jpg")

        result = get_image_path(str(temp_path), "nonexistent")
        assert result is None
