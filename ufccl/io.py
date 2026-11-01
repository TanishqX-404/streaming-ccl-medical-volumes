"""NIfTI and DICOM loading.

Volumes are returned as (Z, Y, X) arrays with voxel spacing (sz, sy, sx) in mm.
NIfTI data are reoriented to the closest canonical (RAS+) frame first, so axis
0 runs inferior->superior, axis 1 posterior->anterior and axis 2 runs from the
patient's left to the patient's right.  Use the same loader for image and mask
so orientations agree.
"""
import os

import numpy as np


def load_nifti(path, canonical=True, dtype=None):
    """dtype: if given, scaled data are read with ``get_fdata`` in that dtype."""
    import nibabel as nib
    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=dtype) if dtype is not None else np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return np.ascontiguousarray(np.transpose(data, (2, 1, 0))), (sz, sy, sx), img.affine


def save_nifti(path, vol_zyx, spacing_zyx=(1.0, 1.0, 1.0)):
    import nibabel as nib
    sz, sy, sx = spacing_zyx
    affine = np.diag([sx, sy, sz, 1.0])
    arr = np.ascontiguousarray(np.transpose(vol_zyx, (2, 1, 0)))
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def load_dicom_series(directory):
    """Read a single-series DICOM directory; returns (HU volume, spacing)."""
    import pydicom
    files = []
    for name in sorted(os.listdir(directory)):
        p = os.path.join(directory, name)
        if not os.path.isfile(p):
            continue
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue
        if hasattr(ds, "PixelData") and hasattr(ds, "ImagePositionPatient"):
            files.append(ds)
    if not files:
        raise ValueError(f"no DICOM image slices found in {directory}")
    iop = np.array(files[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    files.sort(key=lambda d: float(np.dot(normal, np.array(d.ImagePositionPatient, float))))
    pos = np.array([float(np.dot(normal, np.array(d.ImagePositionPatient, float)))
                    for d in files])
    sz = float(np.median(np.diff(pos))) if len(pos) > 1 else float(
        getattr(files[0], "SliceThickness", 1.0))
    sy, sx = (float(v) for v in files[0].PixelSpacing)
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
                    + float(getattr(d, "RescaleIntercept", 0)) for d in files])
    return vol, (abs(sz), sy, sx)


def hu_window_mask(volume_hu, lo=-200.0, hi=300.0):
    """Soft-tissue Hounsfield window mask [lo, hi]."""
    return (volume_hu >= lo) & (volume_hu <= hi)
