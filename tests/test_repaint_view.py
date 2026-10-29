"""The repainted views: a render of the mesh, recoloured with the character's look.

A view nobody photographed has no true picture — but the mesh rendered through the
fitted camera is the geometry seen from that angle, at exactly the pixels the camera
will paint from, and the front's cut-out supplies the colours. These tests pin the
card: the render sits at the bottom, its pixels and position kept for the crop; the
front reference rides above it; the render's own alpha comes back as the view's matte,
eroded a little so the repaint's silhouette drift stays outside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import repaint_view


def _render(size=(320, 448)) -> Image.Image:
    """A clay render: a grey figure shape on transparent surround."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    figure = Image.new("RGBA", (size[0] // 2, size[1] // 2), (160, 160, 160, 255))
    image.paste(figure, (size[0] // 4, size[1] // 4))
    return image


def _front() -> Image.Image:
    image = Image.new("RGBA", (120, 160), (0, 0, 0, 0))
    image.paste(Image.new("RGBA", (40, 40), (214, 178, 148, 255)), (40, 60))
    return image


def test_the_card_keeps_the_render_at_the_bottom_and_carries_its_alpha(tmp_path):
    render = _render()
    card = repaint_view.reference_card(render, _front())

    header = card.height - render.height
    assert card.width == render.width and header > 0
    assert header % 32 == 0                                 # a size the edit returns unscaled

    render_pixels = np.asarray(card)[header:]
    original = np.asarray(render)
    assert (render_pixels == original).all()                # the render's pixels, kept exactly
    alpha = render_pixels[..., 3]
    assert (alpha[original[..., 3] == 0] == 0).all()        # its matte rides in the card
    strip = np.asarray(card)[:header]
    assert strip[..., 3].min() > 0                          # the strip is opaque
    assert strip[..., :3].max() < 255                       # ...on grey, not white


def test_the_front_reference_sits_in_the_strip(tmp_path):
    card = repaint_view.reference_card(_render(), _front())
    header = round(_render().height * 0.3 / 32) * 32
    strip = np.asarray(card.convert("RGB"))[:header]
    centre = strip[:, strip.shape[1] // 3: 2 * strip.shape[1] // 3]
    skin = (np.abs(centre.astype(int) - (214, 178, 148)).max(-1) < 24).sum()
    assert skin > 50                                        # the cut-out is visible in the strip


def test_the_view_is_the_cards_lower_region_matted_by_the_renders_alpha():
    render = _render()
    card = repaint_view.reference_card(render, _front())
    # a repaint that shifts colours but keeps composition, and drifts its silhouette out
    painted = card.convert("RGB")
    header = card.height - render.height
    splat = Image.new("RGBA", card.size, (255, 0, 0, 255))
    painted = Image.alpha_composite(card.convert("RGBA"), splat)
    alpha = card.getchannel("A").crop((0, header, card.width, card.height))

    view = repaint_view.render_from_card(painted, render.size, alpha)

    assert view.size == render.size
    pixels = np.asarray(view)
    assert (pixels[..., :3] == (255, 0, 0)).all()           # the repaint's colour came through
    figure = np.asarray(render)[..., 3] == 255
    eroded = pixels[..., 3] == 255
    assert eroded.sum() <= figure.sum()                     # eroded: drift stays outside
    assert eroded.sum() > figure.sum() * 0.8                # but the figure still mostly counts
    assert (pixels[~eroded, 3] == 0).all()                  # and nothing else is matted in


def test_the_prompt_pins_the_two_jobs_and_names_the_angle():
    words = repaint_view.REPAINT_PROMPT.format(view=repaint_view.VIEW_WORDS["back"])
    assert "from directly behind" in words
    assert "colour reference" in words and "never its shape" in words
    for angle in ("left", "right"):
        assert repaint_view.VIEW_WORDS[angle]
