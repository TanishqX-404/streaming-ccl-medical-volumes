"""Synthetic masks for tests, benchmarks and demos (no patient data)."""
import numpy as np


def random_mask(shape, p, seed=0):
    return np.random.default_rng(seed).random(shape) < p


def checkerboard(shape):
    z, y, x = np.indices(shape)
    return ((z + y + x) % 2) == 0


def snake2d(Y, X):
    """Serpentine path: even rows are corridors, odd rows are walls with one
    gap alternating between the right and left edge."""
    m = np.zeros((Y, X), bool)
    m[0::2, :] = True
    for y in range(1, Y, 2):
        m[y, X - 1 if (y // 2) % 2 == 0 else 0] = True
    return m


def comb(W, H=3):
    """(1, H, W) mask: vertical teeth on even columns joined by a bottom bar.

    Every tooth gets its own provisional label and the bar then merges them one
    by one, which builds a chain of length ~W/2 under naive linking.
    """
    m = np.zeros((1, H, W), bool)
    m[0, :H - 1, 0::2] = True
    m[0, H - 1, :] = True
    return m


def snake3d(Z, Y, X):
    """One long 6-connected path: serpentine layers on even z, joined by single
    voxels on odd z (Y must be odd so layers start and end on corridors)."""
    layer = snake2d(Y, X)
    # the path starts at (0, 0) and ends at the last corridor row's far end
    n_gaps = (Y - 1) // 2
    end_x = X - 1 if n_gaps % 2 == 0 else 0
    ends = [(0, 0), (Y - 1, end_x)]
    vol = np.zeros((Z, Y, X), bool)
    for z in range(0, Z, 2):
        vol[z] = layer
    for z in range(1, Z, 2):
        if z + 1 < Z:
            yy, xx = ends[1] if (z // 2) % 2 == 0 else ends[0]
            vol[z, yy, xx] = True
    return vol


def _value_noise_planes(shape, cell, seed):
    """Yield float32 (Y, X) slices of trilinearly interpolated value noise."""
    Z, Y, X = shape
    ny, nx = Y // cell + 2, X // cell + 2
    yi = np.arange(Y) / cell
    y0 = yi.astype(np.int64)
    wy = (yi - y0).astype(np.float32)[:, None]
    xi = np.arange(X) / cell
    x0 = xi.astype(np.int64)
    wx = (xi - x0).astype(np.float32)[None, :]

    def plane(k):
        c = np.random.default_rng([seed, cell, k]).random((ny, nx), dtype=np.float32)
        r = c[y0] * (1 - wy) + c[y0 + 1] * wy
        return r[:, x0] * (1 - wx) + r[:, x0 + 1] * wx

    pa, pb, k = plane(0), plane(1), 0
    for z in range(Z):
        kz = z // cell
        if kz != k:
            pa, pb, k = pb, plane(kz + 1), kz
        t = np.float32((z % cell) / cell)
        yield (1 - t) * pa + t * pb


def blob_slices(shape, cell=16, thr=0.62, seed=0):
    """Smooth organic blobs: two octaves of value noise, thresholded."""
    a = _value_noise_planes(shape, cell, seed)
    b = _value_noise_planes(shape, max(cell // 4, 2), seed + 1)
    for pa, pb in zip(a, b):
        yield (pa + np.float32(0.5) * pb) / np.float32(1.5) > thr


def noise_slices(shape, p, seed=0):
    Z, Y, X = shape
    rng = np.random.default_rng(seed)
    for _ in range(Z):
        yield rng.random((Y, X), dtype=np.float32) < p


def stack(slices):
    return np.stack(list(slices))


def lesion_phantom(shape, n_lesions=30, r_range=(3, 12), n_vessels=0, seed=0):
    """Sparse ellipsoidal lesions (and optional random-walk vessels).

    Returns a bool mask.  Lesions may overlap and merge, so the number of
    connected components is generally a little below ``n_lesions``.
    """
    Z, Y, X = shape
    rng = np.random.default_rng(seed)
    m = np.zeros(shape, bool)

    def draw_ellipsoid(c, r):
        lo = np.maximum(np.floor(c - r).astype(int), 0)
        hi = np.minimum(np.ceil(c + r).astype(int) + 1, [Z, Y, X])
        if np.any(hi <= lo):
            return
        zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2
        m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= d <= 1.0

    for _ in range(n_lesions):
        c = rng.random(3) * [Z, Y, X]
        r = rng.uniform(*r_range, size=3)
        draw_ellipsoid(c, r)
    for _ in range(n_vessels):
        p = rng.random(3) * [Z, Y, X]
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for _ in range(max(Y, X) // 2):
            draw_ellipsoid(p, np.array([1.5, 1.5, 1.5]))
            d = d + 0.3 * rng.normal(size=3)
            d /= np.linalg.norm(d)
            p = p + 2.0 * d
            if np.any(p < 0) or np.any(p >= [Z, Y, X]):
                break
    return m
