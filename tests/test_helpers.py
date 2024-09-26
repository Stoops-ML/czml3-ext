import pytest

from czml3_ext.borders import available_billboards, available_borders
from czml3_ext.errors import BillboardNotFound, BorderNotFound
from czml3_ext.helpers import get_billboard, get_border


def test_billboard_not_found():
    with pytest.raises(BillboardNotFound):
        _ = get_billboard("F21")
    with pytest.raises(BillboardNotFound):
        _ = get_billboard("F22.csv")


def test_border_not_found():
    with pytest.raises(BorderNotFound):
        _ = get_border("F21")
    with pytest.raises(BorderNotFound):
        _ = get_border("F22.csv")


def test_get_border():
    for b in available_borders:
        get_border(b)


def test_get_billboard():
    for b in available_billboards:
        get_billboard(b)
