import pytest

from czml3_ext.colours import (
    RGBA,
    RGBA_black,
    RGBA_blue,
    RGBA_green,
    RGBA_grey,
    RGBA_orange,
    RGBA_pink,
    RGBA_purple,
    RGBA_red,
    RGBA_white,
    RGBA_yellow,
    create_palette,
)


def test_values():
    assert RGBA_white == RGBA(255.0, 255.0, 255.0, 250.0)
    assert RGBA_red == RGBA(255.0, 0.0, 0.0, 250.0)
    assert RGBA_blue == RGBA(0.0, 0.0, 255.0, 250.0)
    assert RGBA_green == RGBA(0.0, 255.0, 0.0, 250.0)
    assert RGBA_yellow == RGBA(255.0, 255.0, 0.0, 250.0)
    assert RGBA_grey == RGBA(128.0, 128.0, 128.0, 250.0)
    assert RGBA_black == RGBA(0.0, 0.0, 0.0, 255.0)
    assert RGBA_pink == RGBA(255.0, 0.0, 255.0, 255.0)
    assert RGBA_orange == RGBA(255.0, 128.0, 0.0, 255.0)
    assert RGBA_purple == RGBA(127.0, 0.0, 255.0, 255.0)


@pytest.mark.parametrize(
    "c",
    [
        RGBA_white,
        RGBA_red,
        RGBA_blue,
        RGBA_green,
        RGBA_yellow,
        RGBA_grey,
        RGBA_black,
        RGBA_pink,
        RGBA_orange,
        RGBA_purple,
    ],
)
def test_get_with_temp_alpha(c: RGBA):
    alpha = 10
    new = c.get_with_temp_alpha(alpha)
    assert new[3] == alpha
    with pytest.raises(TypeError):
        c[3] = alpha


@pytest.mark.parametrize(
    "c",
    [
        RGBA_white,
        RGBA_red,
        RGBA_blue,
        RGBA_green,
        RGBA_yellow,
        RGBA_grey,
        RGBA_black,
        RGBA_pink,
        RGBA_orange,
        RGBA_purple,
    ],
)
def test_copy(c: RGBA):
    alpha = 10.0
    with pytest.raises(TypeError):
        c[3] = alpha
    new = c.copy()
    assert isinstance(new, RGBA)
    assert new == c
    new[3] = alpha
    assert new[3] == alpha


def test_errors_of_RGBA():
    with pytest.raises(ValueError):
        RGBA(-1, 1, 1, 1)  # outside of bounds
    with pytest.raises(ValueError):
        RGBA(1, 1, 1, 1, 1)  # too many values
    with pytest.raises(ValueError):
        RGBA("1", 1, 1, 1)  # wrong type
    with pytest.raises(ValueError):
        RGBA(1, 1, 1, 1) + RGBA(1, 1, 1, 1)
    with pytest.raises(ValueError):
        RGBA(1, 1, 1, 1).append(RGBA(1, 1, 1, 1))
    with pytest.raises(ValueError):
        RGBA(1, 1, 1, 1).extend(RGBA(1, 1, 1, 1))


@pytest.mark.parametrize(
    "c",
    [
        RGBA_white,
        RGBA_red,
        RGBA_blue,
        RGBA_green,
        RGBA_yellow,
        RGBA_grey,
        RGBA_black,
        RGBA_pink,
        RGBA_orange,
        RGBA_purple,
    ],
)
def test_errors_of_RGBA_copy(c: RGBA):
    new = c.copy()
    with pytest.raises(ValueError):
        new[0] = -1  # outside of bounds
    with pytest.raises(IndexError):
        new[4] = 1.0  # index doesn't exist
    with pytest.raises(ValueError):  # wrong type
        new[0] = "1"  # type: ignore[call-overload]


def test_create_palette():
    assert create_palette([RGBA_black, RGBA_white], 10) == [
        [0.0, 0.0, 0.0, 255.0],
        [
            28.333333333333332,
            28.333333333333332,
            28.333333333333332,
            254.44444444444446,
        ],
        [
            56.666666666666664,
            56.666666666666664,
            56.666666666666664,
            253.88888888888889,
        ],
        [85.0, 85.0, 85.0, 253.33333333333334],
        [
            113.33333333333333,
            113.33333333333333,
            113.33333333333333,
            252.77777777777777,
        ],
        [
            141.66666666666666,
            141.66666666666666,
            141.66666666666666,
            252.22222222222223,
        ],
        [170.0, 170.0, 170.0, 251.66666666666666],
        [
            198.33333333333331,
            198.33333333333331,
            198.33333333333331,
            251.11111111111111,
        ],
        [
            226.66666666666666,
            226.66666666666666,
            226.66666666666666,
            250.55555555555554,
        ],
        [255.0, 255.0, 255.0, 250.0],
    ]
