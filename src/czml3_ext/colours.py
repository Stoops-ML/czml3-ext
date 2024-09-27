from collections.abc import Sequence

import numpy as np

COLOUR_TYPE = list[float]


class RGBA(COLOUR_TYPE):
    """
    A collection of colours in rgba format
    """

    white = [255.0, 255.0, 255.0, 250.0]
    red = [255.0, 0.0, 0.0, 250.0]
    blue = [0.0, 0.0, 255.0, 250.0]
    green = [0.0, 255.0, 0.0, 250.0]
    yellow = [255.0, 255.0, 0.0, 250.0]
    grey = [128.0, 128.0, 128.0, 250.0]
    black = [0.0, 0.0, 0.0, 255.0]
    pink = [255.0, 0.0, 255.0, 255.0]
    orange = [255.0, 128.0, 0.0, 255.0]
    purple = [127.0, 0.0, 255.0, 255.0]


def create_palette(colours: Sequence[COLOUR_TYPE], num_steps: int) -> list[COLOUR_TYPE]:
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

    out: list[COLOUR_TYPE] = []
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
            np.linspace(colours[i], colours[i + 1], num=i_end - i_start).tolist()
        )
    return out
