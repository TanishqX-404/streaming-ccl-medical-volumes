"""Block-addressable volume backends: RAM arrays and raw files on disk.

Both expose ``shape``, ``read_block(z0, z1, y0, y1, x0, x1)`` and
``write_block(z0, y0, x0, block)`` so the tiled labeler never needs the whole
volume in memory.  Raw files use explicit reads and writes (not memory maps) so
resident memory reflects what the algorithm actually holds.
"""
import numpy as np


class ArrayVolume:
    def __init__(self, arr):
        self.arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype

    def read_block(self, z0, z1, y0, y1, x0, x1):
        return self.arr[z0:z1, y0:y1, x0:x1]

    def write_block(self, z0, y0, x0, block):
        z1, y1, x1 = z0 + block.shape[0], y0 + block.shape[1], x0 + block.shape[2]
        self.arr[z0:z1, y0:y1, x0:x1] = block

    def slices(self):
        for z in range(self.shape[0]):
            yield self.arr[z]

    def close(self):
        pass


class RawVolume:
    """C-ordered (Z, Y, X) volume stored as a headerless binary file."""

    def __init__(self, path, shape, dtype=np.uint8, mode="r"):
        self.path = str(path)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.f = open(self.path, "rb" if mode == "r" else "r+b")

    @classmethod
    def create(cls, path, shape, dtype=np.uint8):
        """Zero-filled file of the right size, opened read/write."""
        nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        with open(path, "wb") as f:
            f.truncate(nbytes)
        return cls(path, shape, dtype, mode="rw")

    @classmethod
    def from_slices(cls, path, slices, shape_yx, dtype=np.uint8):
        """Write an iterable of 2-D slices to ``path``; returns the volume."""
        n = 0
        with open(path, "wb") as f:
            for s in slices:
                f.write(np.ascontiguousarray(s, dtype=dtype).tobytes())
                n += 1
        return cls(path, (n,) + tuple(shape_yx), dtype, mode="r")

    def read_block(self, z0, z1, y0, y1, x0, x1):
        _, Y, X = self.shape
        isz = self.dtype.itemsize
        rows = y1 - y0
        out = np.empty((z1 - z0, rows, x1 - x0), self.dtype)
        for i, z in enumerate(range(z0, z1)):
            self.f.seek(((z * Y + y0) * X) * isz)
            buf = self.f.read(rows * X * isz)
            out[i] = np.frombuffer(buf, self.dtype).reshape(rows, X)[:, x0:x1]
        return out

    def write_block(self, z0, y0, x0, block):
        _, Y, X = self.shape
        isz = self.dtype.itemsize
        rows = block.shape[1]
        full_width = block.shape[2] == X
        for i in range(block.shape[0]):
            off = (((z0 + i) * Y + y0) * X) * isz
            if full_width:
                data = np.ascontiguousarray(block[i], dtype=self.dtype)
            else:
                self.f.seek(off)
                data = np.frombuffer(self.f.read(rows * X * isz), self.dtype)
                data = data.reshape(rows, X).copy()
                data[:, x0:x0 + block.shape[2]] = block[i]
            self.f.seek(off)
            self.f.write(data.tobytes())

    def slices(self):
        Z, Y, X = self.shape
        isz = self.dtype.itemsize
        self.f.seek(0)
        for _ in range(Z):
            yield np.frombuffer(self.f.read(Y * X * isz), self.dtype).reshape(Y, X)

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()
