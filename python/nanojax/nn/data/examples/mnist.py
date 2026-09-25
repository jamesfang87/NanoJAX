from __future__ import annotations

import gzip
import shutil
import struct
import urllib.request
from pathlib import Path

from .... import _tensor as nj
from ..dataset import Dataset, Sample

_MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_IMAGE_MAGIC = 0x00000803
_LABEL_MAGIC = 0x00000801


def _read_images(path: Path) -> tuple[bytes, int, int, int]:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != _IMAGE_MAGIC:
            raise ValueError(f"MNISTDataset: invalid images file magic number: {path}")
        data = stream.read(count * rows * cols)
    return data, count, rows, cols


def _read_labels(path: Path) -> bytes:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != _LABEL_MAGIC:
            raise ValueError(f"MNISTDataset: invalid labels file magic number: {path}")
        return stream.read(count)


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    with urllib.request.urlopen(url) as response, open(destination, "wb") as sink:
        shutil.copyfileobj(response, sink)


class MNISTDataset(Dataset):
    rows = 28
    cols = 28
    image_size = rows * cols
    num_classes = 10

    def __init__(self, root: str = "data", train: bool = True, download: bool = True) -> None:
        root = Path(root)
        prefix = "train" if train else "t10k"
        images_path = root / f"{prefix}-images-idx3-ubyte.gz"
        labels_path = root / f"{prefix}-labels-idx1-ubyte.gz"

        if download:
            root.mkdir(parents=True, exist_ok=True)
            _download(_MNIST_URL + images_path.name, images_path)
            _download(_MNIST_URL + labels_path.name, labels_path)

        images, count, rows, cols = _read_images(images_path)
        labels = _read_labels(labels_path)

        if rows != self.rows or cols != self.cols:
            raise ValueError("MNISTDataset: unexpected image dimensions")
        if len(labels) != count:
            raise ValueError("MNISTDataset: image/label count mismatch")

        self._images = images
        self._labels = labels
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError("MNISTDataset: index out of range")

        start = index * self.image_size
        pixels = self._images[start : start + self.image_size]
        x = nj.Tensor([pixel / 255.0 for pixel in pixels])

        y = nj.zeros([self.num_classes])
        y[self._labels[index]] = 1.0

        return Sample(x, y)
