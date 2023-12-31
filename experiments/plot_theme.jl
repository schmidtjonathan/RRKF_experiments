using CairoMakie
using ColorSchemes

DOTS_PER_INCH = 72.0
GOLDEN_RATIO = (5.0^0.5 - 1.0) / 2.0
FULL_WIDTH = 5.5 * DOTS_PER_INCH
FULL_HEIGHT = FULL_WIDTH * GOLDEN_RATIO / 2
HALF_WIDTH = 5.5 / 2 * DOTS_PER_INCH
HALF_HEIGHT = HALF_WIDTH * GOLDEN_RATIO

COLORS = ColorSchemes.tableau_10.colors[[1, 2, 5, 7]]

PN_COLORS =
    parse.(
        RGBf,
        [
            "#107D79",
            "#FF9933",
            "#1F77B4",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
            "#7F7F7F",
            "#BCBD22",
            "#17BECF",
        ],
    )

# Gruvbox
_COLORS =
    parse.(
        RGBf,
        [
            "#a54242", # red
            "#8c9440", # green
            "#de935f", # yellow
            "#5f819d", # blue
            "#85678f", # purple
            "#5e8d87", # aqua
            "#707880", # white
        ],
    )

# Other gruvbox: https://camo.githubusercontent.com/410b3ab80570bcd5b470a08d84f93caa5b4962ccd994ebceeb3d1f78364c2120/687474703a2f2f692e696d6775722e636f6d2f776136363678672e706e67
GRUVBOX_DARK =
    parse.(
        RGBf,
        [
            "#cc241d", # red
            "#98971a", # green
            "#d79921", # yellow
            "#458588", # blue
            "#b16286", # purple
            "#689d6a", # aqua
            "#d65d0e", # orange
        ],
    )
# Lighter colors
GRUVBOX_LIGHT =
    parse.(
        RGBf,
        [
            "#fb4934", # red
            "#b8bb26", # green
            "#fabd2f", # yellow
            "#83a598", # blue
            "#d3869b", # purple
            "#8ec07c", # aqua
            "#fe8019", # orange
        ],
    )

set_theme!()
fontsize_offset = 0
BASE_FONTSIZE = 10
theme = Theme(
    fontsize=BASE_FONTSIZE + fontsize_offset,
    # Figure=(figure_padding = 5),
    Axis=(
        xgridvisible=false,
        ygridvisible=false,
        topspinevisible=false,
        rightspinevisible=false,
        spinewidth=0.5,
        xtickwidth=0.5,
        ytickwidth=0.5,
        xticksize=2,
        yticksize=2,
        xticklabelsize=BASE_FONTSIZE - 3,
        yticklabelsize=BASE_FONTSIZE - 3,
        xlabelsize=BASE_FONTSIZE - 1,
        ylabelsize=BASE_FONTSIZE - 1,
        titlesize=BASE_FONTSIZE - 1,
        xlabelfont="Times New Roman",
        xticklabelfont="Times New Roman regular",
        ylabelfont="Times New Roman",
        yticklabelfont="Times New Roman regular",
        titlefont="Times New Roman",
    ),
    Legend=(
        labelsize=BASE_FONTSIZE - 1,
        framevisible=false,
        patchsize=(8, 8),
        padding=(0, 0, 0, 0),
        labelfont="Times New Roman",
        # colgap=0.0,
        rowgap=1,
    ),
    Scatter=(strokewidth=0.2, markersize=5),
    QQPair=(strokewidth=0.2, markersize=5),
    ScatterLines=(linewidth=2, strokewidth=0.5, markersize=5),
    # Scene = (
    #     patchstrokewidth=0.1,
    # ),
    Colorbar=(
        spinewidth=0.5,
        tickwidth=0.5,
        # ytickwidth=0.5,
        labelsize=BASE_FONTSIZE - 1,
        ticklabelsize=BASE_FONTSIZE - 3,
    ),
)
set_theme!(theme)
