import numpy as np
import pytest
import scipy.ndimage as ndi

STRUCT = {6: 1, 18: 2, 26: 3}


def scipy_label(mask, conn):
    return ndi.label(mask, ndi.generate_binary_structure(3, STRUCT[conn]))


@pytest.fixture(params=[6, 18, 26])
def conn(request):
    return request.param
