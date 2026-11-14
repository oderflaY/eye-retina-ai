"""Pruebas de detección de formato y lectura de etiquetas."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.eye_labels import (
    AmbiguousLabelFormat,
    LabelFormatError,
    detect_format,
    load_eye_samples,
)
from src.data.synthetic import generate


def _img(path: Path, w: int = 200, h: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), np.uint8))


def test_csv_synthetic(tmp_path: Path) -> None:
    generate(tmp_path, n=12, seed=1)
    assert detect_format(tmp_path) == "csv"
    samples = load_eye_samples(tmp_path)
    assert len(samples) == 12
    with_eyes = [s for s in samples if s.num_eyes]
    assert with_eyes and all(s.num_eyes == 2 for s in with_eyes)
    assert not np.isnan(with_eyes[0].landmarks).any()


def test_coco(tmp_path: Path) -> None:
    _img(tmp_path / "imgs" / "a.jpg")
    kps = []
    for k in range(6):
        kps += [20 + k, 30, 2 if k != 3 else 0]
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 20, 40, 20], "keypoints": kps}
        ],
        "categories": [{"id": 3, "name": "eye"}, {"id": 4, "name": "nose"}],
    }
    (tmp_path / "imgs" / "ann.json").write_text(json.dumps(coco))
    assert detect_format(tmp_path) == "coco"
    (s,) = load_eye_samples(tmp_path)
    np.testing.assert_allclose(s.boxes[0], [10, 20, 40, 20])
    assert np.isnan(s.landmarks[0, 3]).all() and not np.isnan(s.landmarks[0, 0]).any()


def test_yolo_normalized(tmp_path: Path) -> None:
    _img(tmp_path / "images" / "b.png", 200, 100)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert detect_format(tmp_path) == "yolo"
    (s,) = load_eye_samples(tmp_path)
    np.testing.assert_allclose(s.boxes[0], [80, 40, 40, 20], atol=1e-4)


def test_voc(tmp_path: Path) -> None:
    _img(tmp_path / "c.jpg")
    (tmp_path / "c.xml").write_text(
        "<annotation><filename>c.jpg</filename>"
        "<object><name>eye</name><bndbox><xmin>5</xmin><ymin>6</ymin><xmax>25</xmax><ymax>16</ymax></bndbox></object>"
        "<object><name>mouth</name><bndbox><xmin>50</xmin><ymin>60</ymin>"
        "<xmax>90</xmax><ymax>80</ymax></bndbox></object>"
        "</annotation>"
    )
    (s,) = load_eye_samples(tmp_path)
    assert s.num_eyes == 1
    np.testing.assert_allclose(s.boxes[0], [5, 6, 20, 10])


def test_ambiguous_formats(tmp_path: Path) -> None:
    generate(tmp_path, n=3, seed=2)
    (tmp_path / "x.xml").write_text("<annotation><filename>images/synth_0000.png</filename></annotation>")
    with pytest.raises(AmbiguousLabelFormat):
        detect_format(tmp_path)
    # Con el formato explícito sí se puede leer
    assert len(load_eye_samples(tmp_path, "csv")) == 3


def test_no_labels(tmp_path: Path) -> None:
    _img(tmp_path / "solo.jpg")
    with pytest.raises(LabelFormatError):
        detect_format(tmp_path)


def test_csv_unknown_classes(tmp_path: Path) -> None:
    _img(tmp_path / "d.jpg")
    (tmp_path / "ann.csv").write_text("image,class,x,y,w,h\nd.jpg,cat,1,1,10,10\nd.jpg,dog,20,20,10,10\n")
    with pytest.raises(AmbiguousLabelFormat):
        load_eye_samples(tmp_path)
