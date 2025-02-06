import numpy as np
import matplotlib.pyplot as plt

madcmap = "viridis"
num_total = 25


def rgb_to_greenblind(clr):
    clr = clr * 255
    new_clr = np.zeros(3)
    new_clr[:2] = (4211 + 0.677 * clr[1] ** 2.2 + 0.2802 * clr[0] ** 2.2) ** (1 / 2.2)
    new_clr[2] = (
        4211
        + 0.95724 * clr[2] ** 2.2
        + 0.02138 * clr[1] ** 2.2
        - 0.02138 * clr[0] ** 2.2
    ) ** (1 / 2.2)
    return new_clr / 255


def interp_colors(clr_min, clr_max, num_total: int = 5):
    alph = np.linspace(0, 1, num_total)[1:-1]
    return (
        [clr_min]
        + list(clr_min * (1 - alph[:, None]) + clr_max * alph[:, None])
        + [clr_max]
    )


def madimshow(
    mat,
    cmap: str = madcmap,
    xlabel: str = "",
    ylabel: str = "",
    axis=True,
    figsize=(4, 4),
    vmin=None,
    vmax=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    im_args = {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    ax.imshow(mat, **{k: v for k, v in im_args.items() if v is not None})
    ax.set_xlabel(xlabel) if xlabel else None
    ax.set_ylabel(ylabel) if ylabel else None
    ax.axis("off") if not axis else None
    fig.tight_layout()


color_schemes = {
    "bright_qual": {
        "blue": [68, 119, 170],
        "cyan": [102, 204, 238],
        "green": [34, 136, 51],
        "yellow": [204, 187, 68],
        "red": [238, 102, 119],
        "purple": [170, 51, 119],
        "gray": [187, 187, 187],
    },
    "highcont_qual": {
        "yellow": [221, 170, 51],
        "red": [187, 85, 102],
        "blue": [0, 68, 136],
    },
    "vib_qual": {
        "blue": [0, 119, 187],
        "cyan": [51, 187, 238],
        "teal": [0, 153, 136],
        "orange": [238, 119, 51],
        "red": [204, 51, 17],
        "magenta": [238, 51, 119],
        "gray": [187, 187, 187],
    },
    "muted_qual": {
        "indigo": [51, 34, 136],
        "cyan": [136, 204, 238],
        "teal": [68, 170, 153],
        "green": [17, 119, 51],
        "olive": [153, 153, 51],
        "sand": [221, 204, 119],
        "rose": [204, 102, 119],
        "wine": [136, 34, 85],
        "purple": [170, 68, 153],
        "gray": [221, 221, 221],
    },
    "medcont_qual": {
        "light_yellow": [238, 204, 102],
        "light_red": [238, 153, 170],
        "light_blue": [102, 153, 204],
        "dark_yellow": [153, 119, 0],
        "dark_red": [153, 68, 85],
        "dark_blue": [0, 68, 136],
    },
    "pale_qual": {
        "pale_blue": [187, 204, 238],
        "pale_cyan": [204, 238, 255],
        "pale_green": [204, 221, 170],
        "pale_yellow": [238, 238, 187],
        "pale_red": [255, 204, 204],
        "pale_gray": [221, 221, 221],
        "dark_blue": [34, 34, 85],
        "dark_cyan": [34, 85, 85],
        "dark_green": [34, 85, 34],
        "dark_yellow": [102, 102, 51],
        "dark_red": [102, 51, 51],
        "dark_gray": [85, 85, 85],
    },
    "light_qual": {
        "blue": [119, 170, 221],
        "cyan": [153, 221, 255],
        "mint": [68, 187, 153],
        "pear": [187, 204, 51],
        "olive": [170, 170, 0],
        "yellow": [238, 221, 136],
        "orange": [238, 136, 102],
        "pink": [255, 170, 187],
        "gray": [221, 221, 221],
    },
}

for scheme in color_schemes:
    for key in color_schemes[scheme]:
        color_schemes[scheme][key] = np.array(color_schemes[scheme][key]) / 255
