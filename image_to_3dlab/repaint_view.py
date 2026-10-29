"""Repaint a render of the mesh with the character's real look, keeping its structure.

A view nobody photographed (the back, the sides) has no true picture — but the mesh can
be rendered through the fitted camera, and that render *is* the geometry seen from that
angle: an apron in the mesh is an apron in the render, at exactly the pixels the camera
will paint from. What the render lacks is colour, which the front picture has. So the
repaint is an edit with two jobs shown in one reference image: the render, large,
framed exactly as the camera sees it, and the front's cut-out as a small colour
reference above it. The edit keeps composition (the same property the back-view draw
relies on), so the lower region comes back with its pose, proportions and framing
untouched — cropped out, it is a picture of this mesh from this camera, in the
character's colours.

A freely *drawn* view has no such anchor: Qwen invents the back it thinks the character
should have — crossed apron straps on a mesh whose back is a full apron (2026-10-28) —
and the paint then disagrees with the geometry underneath.
"""

from __future__ import annotations

from PIL import Image, ImageFilter

#: The share of the card's height the front reference strip takes.
HEADER_SHARE = 0.3

#: Background of the card, the plain light grey the drawn-back reference uses.
CARD_GREY = (238, 238, 238)

#: The repaint's prompt. `{view}` names the angle; everything else pins the two jobs:
#: colour the render from the reference above, never reshape it.
REPAINT_PROMPT = (
    "The lower part of this image is a plain grey 3D render of a character seen "
    "{view}. The small figure in the strip above it is the same character seen from "
    "the front, as a colour reference. Repaint the grey render in the lower part with "
    "the character's real appearance from the reference: the same colours, materials "
    "and clothing designs, flat even lighting, plain light grey background. Keep the "
    "grey render's exact pose, proportions, framing and position unchanged — change "
    "its colours only, never its shape. Leave the small reference figure unchanged."
)

#: How each angle is named in the prompt.
VIEW_WORDS = {
    "back": "from directly behind",
    "left": "from its left side",
    "right": "from its right side",
}


def reference_card(render: Image.Image, front: Image.Image) -> Image.Image:
    """The render with the front's cut-out as a small colour reference above it.

    The render sits at the bottom, its exact pixels and position — the part the edit
    must keep. The card is RGBA and the render's own alpha rides along in its lower
    region: the repaint worker's matte, so the view needs no cut-out of its own. The
    strip's height is a multiple of 32 so the whole card is a size the edit returns
    unscaled.
    """
    header = 32 * max(round(render.height * HEADER_SHARE / 32), 1)
    card = Image.new("RGBA", (render.width, render.height + header), (*CARD_GREY, 255))
    thumb_h = header - round(header * 0.25)
    cut = front.convert("RGBA")
    cut.thumbnail((max(round(cut.width * thumb_h / max(cut.height, 1)), 1), thumb_h))
    card.paste(cut, ((card.width - cut.width) // 2, (header - cut.height) // 2), cut)
    card.paste(render.convert("RGBA"), (0, header))
    return card


def render_from_card(card: Image.Image, render_size: tuple[int, int],
                     alpha: Image.Image) -> Image.Image:
    """The repainted view: the card's lower region at `render_size`, matted with `alpha`.

    `alpha` is the render's own alpha channel, eroded a little so the repaint's small
    silhouette drift stays outside the matte.
    """
    _width, height = render_size
    header = card.height - height
    view = card.crop((0, header, card.width, header + height)).convert("RGBA")
    view.putalpha(alpha.filter(ImageFilter.MinFilter(5)))
    return view
