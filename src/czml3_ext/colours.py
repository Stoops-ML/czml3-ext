from collections.abc import Sequence
from typing import Any

import numpy as np


class RGBA(list[float]):
    def __init__(self, *args):
        if not all(isinstance(x, float) for x in args):
            raise ValueError("All elements must be floats")
        if not all(x <= 255 and x >= 0 for x in args):
            raise ValueError("All elements must be between 0 and 255")
        if len(args) != 4:
            raise ValueError("Colour must have four values: RGBA")
        super().__init__(args)
        self._modify = False

    def append(self, item):
        raise TypeError("Appending not allowed: RGBA must have four values.")

    def extend(self, item):
        raise TypeError("Extending not allowed: RGBA must have four values.")

    def __add__(self, item):
        raise TypeError("Extending not allowed: RGBA must have four values.")

    def __setitem__(self, index, item: Any) -> None:
        if not self._modify:
            raise TypeError(
                "Modify flag is disabled. Please use c.copy() to allow modifications."
            )
        if not isinstance(item, float):
            raise ValueError("Only float values can be assigned")
        if item > 255 or item < 0:
            raise ValueError("Alpha value must be between 0 and 255")
        super().__setitem__(index, item)

    def get_with_temp_alpha(self, alpha: float | int) -> list[float]:
        if not isinstance(alpha, float | int):
            raise TypeError("Alpha value must be float or int")
        if alpha > 255 or alpha < 0:
            raise ValueError("Alpha value must be between 0 and 255")
        tmp = self.copy()
        tmp[3] = float(alpha)
        return tmp

    def copy(self) -> list[float]:
        new = RGBA(
            super().__getitem__(0),
            super().__getitem__(1),
            super().__getitem__(2),
            super().__getitem__(3),
        )
        new._modify = True
        return new


RGBA_white = RGBA(255.0, 255.0, 255.0, 250.0)
RGBA_red = RGBA(255.0, 0.0, 0.0, 250.0)
RGBA_blue = RGBA(0.0, 0.0, 255.0, 250.0)
RGBA_green = RGBA(0.0, 255.0, 0.0, 250.0)
RGBA_yellow = RGBA(255.0, 255.0, 0.0, 250.0)
RGBA_grey = RGBA(128.0, 128.0, 128.0, 250.0)
RGBA_black = RGBA(0.0, 0.0, 0.0, 255.0)
RGBA_pink = RGBA(255.0, 0.0, 255.0, 255.0)
RGBA_orange = RGBA(255.0, 128.0, 0.0, 255.0)
RGBA_purple = RGBA(127.0, 0.0, 255.0, 255.0)


def create_palette(colours: Sequence[RGBA], num_steps: int) -> list[RGBA]:
    for c in colours:
        assert (
            len(c) == 4
        ), "Each colour in colours must be RGBA (sequence with four elements)."
    num_sections = len(colours) - 1
    assert (
        num_sections > 0
    ), "Number of colours must be greater than one to create a palette"
    num_steps_per_colour = int(np.floor(num_steps / num_sections))
    remainder = (num_steps / num_sections) % 1

    out: list[RGBA] = []
    added = 0
    for i in range(num_sections):
        i_start = num_steps_per_colour * i + added
        i_end = num_steps_per_colour * (i + 1) + added
        if remainder * (i + 1) >= 1:
            i_end += 1
            added += 1
        if i_end > num_steps:
            i_end = num_steps
        out.extend(
            np.linspace(colours[i], colours[i + 1], num=i_end - i_start).tolist()  # type: ignore
        )
    return out
