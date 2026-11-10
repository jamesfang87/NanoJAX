"""Tests for the vectorized MNIST batch path."""

from __future__ import annotations

import pathlib
import unittest

from nanojax.nn.data import DataLoader, MNISTDataset
from nanojax.nn.data.dataloader import _collate

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HAS_MNIST = (DATA / "train-images-idx3-ubyte.gz").exists()


@unittest.skipUnless(HAS_MNIST, "MNIST data files are not present")
class TestMNISTBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = MNISTDataset(root=str(DATA), train=True)

    def test_get_batch_matches_per_sample_path(self):
        indices = [0, 1, 2, 17, 123]
        fast = self.dataset.get_batch(indices)
        slow = _collate([self.dataset[i] for i in indices])
        self.assertEqual(fast.x.shape, slow.x.shape)
        self.assertEqual(fast.y.shape, slow.y.shape)
        self.assertTrue(fast.x.allclose(slow.x, 1e-6))
        self.assertTrue(fast.y.allclose(slow.y, 1e-6))

    def test_loader_uses_vectorized_batches(self):
        loader = DataLoader(self.dataset, 128, shuffle=False)
        batch = next(iter(loader))
        expected = self.dataset.get_batch(list(range(128)))
        self.assertEqual(batch.x.shape, [128, 784])
        self.assertEqual(batch.y.shape, [128, 10])
        self.assertTrue(batch.x.allclose(expected.x, 1e-6))
        self.assertTrue(batch.y.allclose(expected.y, 1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
