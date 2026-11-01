import numpy as np
import pytest

from ufccl.io import hu_window_mask, load_dicom_series, load_nifti, save_nifti


def test_nifti_roundtrip_keeps_axes_and_spacing(tmp_path):
    vol = np.zeros((6, 8, 10), np.uint8)
    vol[1, 2, 3] = 1           # z, y, x
    vol[5, 7, 9] = 1
    save_nifti(tmp_path / "m.nii.gz", vol, (3.0, 0.9, 0.7))
    got, spacing, _ = load_nifti(tmp_path / "m.nii.gz")
    assert got.shape == vol.shape
    assert np.allclose(spacing, (3.0, 0.9, 0.7))
    assert np.array_equal(got, vol)


def test_nifti_reorients_to_canonical(tmp_path):
    import nibabel as nib
    data = np.zeros((4, 5, 6), np.uint8)             # (x, y, z)
    data[0, 0, 0] = 1
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])          # x axis flipped (LAS)
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "las.nii.gz")
    got, _, _ = load_nifti(tmp_path / "las.nii.gz")
    # after RAS+ reorientation the voxel at x=0 moves to the far end of x
    assert got[0, 0, 3] == 1 and got.sum() == 1


def test_dicom_series_hu_and_spacing(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    n, rows, cols = 4, 6, 7
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2000, (n, rows, cols)).astype(np.int16)
    for i in [2, 0, 3, 1]:                           # written out of order
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(tmp_path / f"s{i}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns = rows, cols
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = [0.7, 0.8]
        ds.ImagePositionPatient = [0.0, 0.0, 2.5 * i]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = raw[i].tobytes()
        ds.save_as(str(tmp_path / f"s{i}.dcm"), enforce_file_format=True)
    vol, spacing = load_dicom_series(str(tmp_path))
    assert vol.shape == (n, rows, cols)
    assert np.allclose(spacing, (2.5, 0.7, 0.8))
    assert np.array_equal(vol, raw.astype(np.float32) - 1024)
    assert hu_window_mask(vol, -200, 300).dtype == bool
