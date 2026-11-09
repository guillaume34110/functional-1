"""Original indexed 160×144 artwork and the capture-to-color target palette."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = (
    "#21193b", "#43335d", "#705581", "#a878a0", "#f8dcce", "#fff1d8",
    "#1f284b", "#273454", "#344361", "#465a77", "#8bb3b7", "#c7e7d0",
    "#47326a", "#86569e", "#cd85c0", "#f7b8d0",
    "#744344", "#b65b62", "#ec807c", "#ffc1a1",
    "#705146", "#b17b52", "#f0b765", "#ffe2a2",
    "#235957", "#36847a", "#60bc9a", "#b9e6ba",
    "#334773", "#496fa7", "#80a9dd", "#bed9ff",
    "#613c6a", "#9a558e", "#e48fbb", "#ffcde3",
    "#52456c", "#7068aa", "#aaa3e2", "#dad3ff",
)
PALETTE = [tuple((int(color[i:i + 2], 16) >> 3) * 255 // 31 for i in (1, 3, 5)) for color in COLORS]
FONT = ImageFont.load_default_imagefont()


def lettering(image, x, y, text, ink, scale=1, *, centered=False):
    mask = Image.new("1", (FONT.getlength(text), 11))
    ImageDraw.Draw(mask).text((0, 0), text, font=FONT, fill=1)
    if scale > 1:
        mask = mask.resize((mask.width * scale, mask.height * scale), Image.Resampling.NEAREST)
    image.paste(ink, (x - mask.width // 2 if centered else x, y), mask)


def star(draw, x, y, ink):
    draw.line((x - 2, y, x + 2, y), fill=ink)
    draw.line((x, y - 2, x, y + 2), fill=ink)


def bunny(draw, x, y, body=5):
    draw.rectangle((x + 3, y, x + 7, y + 9), fill=0)
    draw.rectangle((x + 12, y + 1, x + 16, y + 9), fill=0)
    draw.rectangle((x + 4, y + 1, x + 6, y + 9), fill=body)
    draw.rectangle((x + 13, y + 2, x + 15, y + 9), fill=body)
    draw.line((x + 5, y + 3, x + 5, y + 7), fill=14)
    draw.line((x + 14, y + 4, x + 14, y + 7), fill=14)
    draw.rounded_rectangle((x, y + 7, x + 20, y + 21), 5, fill=body, outline=0)
    draw.rectangle((x + 5, y + 13, x + 6, y + 15), fill=0)
    draw.rectangle((x + 14, y + 13, x + 15, y + 15), fill=0)
    draw.point((x + 10, y + 16), fill=32)
    draw.line((x + 3, y + 17, x + 5, y + 17), fill=14)
    draw.line((x + 15, y + 17, x + 17, y + 17), fill=14)


def tile(kind):
    """Four shades per candy tile, with an original glyph for each piece."""
    dark, middle, light, shine = range(12 + 4 * kind, 16 + 4 * kind)
    image = Image.new("P", (8, 8), dark)
    d = ImageDraw.Draw(image)
    d.rectangle((1, 1, 6, 6), fill=middle)
    d.rectangle((1, 1, 6, 1), fill=shine)
    d.line((1, 2, 1, 5), fill=light)
    symbols = (
        ((3, 3), (5, 3), (4, 4), (4, 5)),
        ((4, 2), (3, 3), (5, 3), (4, 4)),
        ((3, 3), (4, 3), (5, 3), (3, 4), (5, 4), (4, 5)),
        ((3, 2), (4, 3), (5, 4), (4, 5)),
        ((4, 2), (3, 3), (5, 3), (3, 5), (5, 5)),
        ((3, 2), (3, 3), (4, 4), (5, 4), (5, 5)),
        ((3, 2), (5, 2), (4, 3), (3, 4), (5, 4)),
    )
    d.point(symbols[kind], fill=shine)
    return np.asarray(image)


def screens():
    game = Image.new("P", (160, 144), 0)
    d = ImageDraw.Draw(game)
    d.rectangle((12, 0, 99, 143), fill=1)
    d.rectangle((15, 0, 96, 143), fill=9)
    d.rectangle((16, 0, 95, 143), fill=6)
    d.rounded_rectangle((101, 3, 158, 55), 5, fill=6, outline=3)
    lettering(game, 130, 4, "NEXT", 5, centered=True)
    star(d, 151, 10, 14)
    for label, y in (("SCORE", 59), ("LINES", 83), ("LEVEL", 107)):
        d.rounded_rectangle((101, y, 158, y + 21), 4, fill=1, outline=2)
        lettering(game, 106, y + 1, label, 4)
    lettering(game, 131, 132, "PROGSOFT", 11, centered=True)
    for y in range(5, 144, 9):
        d.point((4 + (y % 2) * 3, y), fill=3)

    def panel(heading, hint, accent):
        image = Image.new("P", (160, 144), 0)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((5, 5, 154, 138), 10, fill=1, outline=accent, width=2)
        draw.rounded_rectangle((10, 10, 149, 133), 7, fill=6, outline=2)
        for x, y in ((21, 27), (42, 20), (133, 23), (117, 38), (27, 111), (138, 104), (21, 77)):
            star(draw, x, y, accent)
        bunny(draw, 69, 18)
        lettering(image, 81, 52, "NEURONAL", 0, 2, centered=True)
        lettering(image, 80, 50, "NEURONAL", 5, 2, centered=True)
        lettering(image, 81, 74, heading, 0, 2, centered=True)
        lettering(image, 80, 72, heading, accent, 2, centered=True)
        lettering(image, 80, 101, hint, 5, centered=True)
        lettering(image, 80, 118, "PROGSOFT", 11, centered=True)
        return np.asarray(image)

    return {
        "game": np.asarray(game),
        "title": panel("TETRIS", "Z / ENTER TO PLAY", 15),
        "pause": panel("PAUSED", "ENTER TO RESUME", 31),
        "over": panel("GAME OVER", "THANKS FOR PLAYING", 23),
        "credits": panel("CREDITS", "A GAME BY PROGSOFT", 27),
    }
