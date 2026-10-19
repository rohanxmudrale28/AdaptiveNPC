"""Pixel-art drawings for the CONTROLS detail pages (keyboard, keyboard + mouse, controller), drawn with rects and
lines on the 320x240 canvas in the game's palette and font. No image files. Each view highlights the parts that matter
in one colour per action (move / attack / block / dash) and joins them to a label with an arrow of pixel lines."""
import pygame

from adaptive_npc.render import font
from adaptive_npc.render.controller import button_label

HEAD_HALF = (0, 1, 2, 2)           # arrowhead half-width, tip first
LABEL_X = 182                       # label column of the keyboard views
DESC = {"move": "SETS YOUR FACING", "attack": "HITS THE TILE AHEAD", "block": "STOPS FRONT HITS",
        "dash": "2 TILES, CANNOT BE HIT"}
TITLES = {"move": "MOVE", "attack": "ATTACK", "block": "BLOCK (HOLD)", "dash": "DASH"}


def action_color(v, action):
    return v.c["act_" + action]


# ------------------------------------------------------------------ primitives
def arrow(v, pts, color, head=True):
    """Polyline of 1px lines with a start dot and (optionally) an arrowhead on the last, axis-aligned segment."""
    cv = v.canvas
    for a, b in zip(pts, pts[1:]):
        pygame.draw.line(cv, color, a, b)
    sx, sy = pts[0]
    pygame.draw.rect(cv, color, (sx - 1, sy - 1, 3, 3))
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        dx, dy = (x1 > x0) - (x1 < x0), (y1 > y0) - (y1 < y0)
        for k, half in enumerate(HEAD_HALF):
            px, py = x1 - dx * k, y1 - dy * k
            for o in range(-half, half + 1):
                cv.set_at((px - dy * o, py + dx * o), color)


def label(v, name, x, y_c, color, sub=None, right=False):
    """Action label with a dim description under it, vertically centred on y_c."""
    v.text(TITLES[name], x, y_c - 8, color, right=right)
    v.text(sub if sub is not None else DESC[name], x, y_c + 2, v.c["dim"], right=right)


def tri(v, cx, cy, direction, color, n=4):
    """Small filled arrowhead glyph pointing up / down / left / right, used inside arrow keys."""
    for r in range(n):
        if direction in ("up", "down"):
            y = cy - n // 2 + r if direction == "up" else cy + n // 2 - r
            v.box(cx - r, y, 2 * r + 1, 1, color)
        else:
            x = cx - n // 2 + r if direction == "left" else cx + n // 2 - r
            v.box(x, cy - r, 1, 2 * r + 1, color)


def key_cap(v, x, y, name, hot=None, w=None, h=15):
    """A keyboard key (top-left x, y). `name` is a label, or up/down/left/right for an arrow key; `hot` = highlight colour."""
    arrow_key = name in ("up", "down", "left", "right")
    text = "" if arrow_key else name
    w = w or max(15, font.text_width(text) + 7)
    face, edge = (hot, v.c["text"]) if hot else (v.c["wall"], v.c["dim"])
    ink = v.pal["ink"] if hot else v.c["text"]
    v.box(x, y + 2, w, h - 2, v.c["floor_a"])          # the key's side
    v.box(x, y, w, h - 2, face, edge)
    if arrow_key:
        tri(v, x + w // 2, y + (h - 2) // 2, name, ink)
    else:
        v.text(text, x + (w - font.text_width(text)) // 2, y + 3, ink)
    return pygame.Rect(x, y, w, h)


def key_cluster(v, x, y, names, hot):
    """Inverted-T of four keys (up, left, down, right) at pitch 16: 47 x 31 px. `names` = (up, left, down, right) labels."""
    up, left, down, right = names
    key_cap(v, x + 16, y, up, hot)
    key_cap(v, x, y + 16, left, hot)
    key_cap(v, x + 16, y + 16, down, hot)
    key_cap(v, x + 32, y + 16, right, hot)
    return pygame.Rect(x, y, 47, 31)


def pill(v, x, y, w, h, fill, edge, r=3):
    """Rounded (chamfered) block: an outline shape with the fill shape inset by one pixel."""
    for inset, col in ((0, edge), (1, fill)):
        ix, iy, iw, ih, ir = x + inset, y + inset, w - 2 * inset, h - 2 * inset, max(1, r - inset)
        v.box(ix + ir, iy, iw - 2 * ir, ih, col)
        v.box(ix, iy + ir, iw, ih - 2 * ir, col)
        v.box(ix + 1, iy + 1, iw - 2, ih - 2, col)


def disc(v, cx, cy, r, fill, edge=None):
    if edge is not None:
        pygame.draw.circle(v.canvas, edge, (cx, cy), r)
        r -= 1
    pygame.draw.circle(v.canvas, fill, (cx, cy), r)


def pad_indicator(v, x, y, connected, name="", generic=False):
    """Small connected / not-found light with the controller's name (x, y = top-left of the text row).
    `generic` marks a pad without an SDL mapping (it works through the joystick fallback and can be rebound)."""
    disc(v, x + 3, y + 3, 3, v.c["act_move"] if connected else v.c["dim"], v.c["text"] if connected else None)
    txt = ("CONTROLLER CONNECTED: " + name[:24].upper()).rstrip(": ") if connected else "NO CONTROLLER FOUND"
    if connected and generic:
        txt = ("CONNECTED: " + name[:20].upper() + " (GENERIC)")
    v.text(txt, x + 12, y, v.c["text"] if connected else v.c["dim"])


# ------------------------------------------------------------------ shared pieces
def _move_row(v, keys_cfg, y, label_x=LABEL_X):
    """WASD cluster, 'OR', arrow keys, all joined to the MOVE label. Returns the label's y centre."""
    col = action_color(v, "move")
    names = []
    for first in (0, 1):
        names.append([keys_cfg[d][min(first, len(keys_cfg[d]) - 1)] for d in ("up", "left", "down", "right")])
    wasd = key_cluster(v, 24, y, [n.upper() for n in names[0]], col)
    arrows = key_cluster(v, 100, y, [n if n in ("up", "down", "left", "right") else n.upper() for n in names[1]], col)
    v.text("OR", 82, y + 18, v.c["dim"])
    y_c = arrows.centery + 8
    arrow(v, [(arrows.right + 2, y_c), (label_x - 4, y_c)], col)
    arrow(v, [(wasd.centerx, wasd.top - 2), (wasd.centerx, y - 8), (175, y - 8), (175, y_c)], col, head=False)
    label(v, "move", label_x, y_c, col)
    return y_c


def _key_rows(v, keys, rows, ys, x=116):
    """One key per action, each with an arrow to its label. keys = {action: key label}."""
    for action, y_c in zip(rows, ys):
        col = action_color(v, action)
        cap = key_cap(v, x, y_c - 7, keys[action], col)
        arrow(v, [(cap.right + 2, y_c), (LABEL_X - 4, y_c)], col)
        label(v, action, LABEL_X, y_c, col)


# ------------------------------------------------------------------ the three views
def draw_keyboard(v, move_keys, caps):
    """move_keys = the movement bindings (render.yaml `keys`); caps = {attack, block, dash: key label}."""
    _move_row(v, move_keys, 47)
    _key_rows(v, caps, ("attack", "block", "dash"), (102, 130, 158))


def draw_kbm(v, move_keys, caps):
    _move_row(v, move_keys, 47)
    att, blk, dash = (action_color(v, a) for a in ("attack", "block", "dash"))
    # mouse, seen from above: body, two buttons (left / right highlighted), wheel
    mx, my, mw, mh = 103, 88, 40, 52
    pill(v, mx, my, mw, mh, v.c["wall"], v.c["dim"], r=8)
    bw = (mw - 6) // 2
    v.box(mx + 2, my + 2, bw, 22, att, v.c["text"])
    v.box(mx + mw - 2 - bw, my + 2, bw, 22, blk, v.c["text"])
    v.box(mx + mw // 2 - 1, my + 5, 3, 9, v.c["dim"])
    v.text("L", mx + 2 + (bw - 5) // 2, my + 9, v.pal["ink"])
    v.text("R", mx + mw - 2 - bw + (bw - 5) // 2, my + 9, v.pal["ink"])
    arrow(v, [(mx + 2 + bw // 2, my + 1), (mx + 2 + bw // 2, 84), (170, 84), (170, 102), (LABEL_X - 4, 102)], att)
    arrow(v, [(mx + mw - 3, my + 10), (160, my + 10), (160, 130), (LABEL_X - 4, 130)], blk)
    label(v, "attack", LABEL_X, 102, att)
    label(v, "block", LABEL_X, 130, blk)
    cap = key_cap(v, 116, 151, caps["dash"], dash)
    arrow(v, [(cap.right + 2, 158), (LABEL_X - 4, 158)], dash)
    label(v, "dash", LABEL_X, 158, dash)


def draw_controller(v, pad_cfg, connected, name, generic=False, note=None):
    cx, cy = 96, 116
    move, att, blk, dash = (action_color(v, a) for a in ("move", "attack", "block", "dash"))
    body, edge, dim, ink = v.c["wall"], v.c["dim"], v.c["dim"], v.pal["ink"]
    # shoulders, grips, body (grips first so the body overlaps their top)
    for sx in (-40, 18):
        v.box(cx + sx, cy - 31, 22, 5, body, edge)
    for sign in (-1, 1):
        pts = [(cx + sign * 52, cy + 4), (cx + sign * 24, cy + 4), (cx + sign * 33, cy + 42), (cx + sign * 45, cy + 42),
               (cx + sign * 56, cy + 24)]
        pygame.draw.polygon(v.canvas, body, pts)
        pygame.draw.polygon(v.canvas, edge, pts, 1)
    pill(v, cx - 54, cy - 27, 108, 44, body, edge, r=10)
    # left stick (moves), D-pad (also moves), right stick (unused)
    lx, ly = cx - 30, cy - 8
    disc(v, lx, ly, 9, v.c["floor_a"], edge)
    disc(v, lx, ly, 5, move, v.c["text"])
    dx, dy = cx - 14, cy + 8
    v.box(dx - 7, dy - 2, 15, 5, move)
    v.box(dx - 2, dy - 7, 5, 15, move)
    v.box(dx - 1, dy - 1, 3, 3, ink)
    disc(v, cx + 10, cy + 8, 7, v.c["floor_a"], edge)
    disc(v, cx + 10, cy + 8, 3, v.c["wall"], None)
    # face buttons: Y north, B east, A south (by position), X west
    pad_names = {"y": (cx + 30, cy - 19, dash), "b": (cx + 41, cy - 8, blk), "a": (cx + 30, cy + 3, att),
                 "x": (cx + 19, cy - 8, None)}
    for btn, (bx, by, col) in pad_names.items():
        text = button_label(pad_cfg, btn)
        text = text if len(text) <= 2 else ""
        disc(v, bx, by, 6, col if col else v.c["floor_a"], v.c["text"] if col else edge)
        if text:
            v.text(text, bx - font.text_width(text) // 2, by - 3, ink if col else v.c["dim"])
    v.box(cx - 9, cy - 14, 6, 3, edge)                      # back
    v.box(cx + 3, cy - 14, 6, 3, v.c["text"])               # start
    # arrows to the labels
    row_move, row_dash, row_block, row_attack = 60, 84, 108, 132
    arrow(v, [(lx, ly - 9), (lx, row_move), (LABEL_X - 4, row_move)], move)
    arrow(v, [(cx + 36, cy - 19), (162, cy - 19), (162, row_dash), (LABEL_X - 4, row_dash)], dash)
    arrow(v, [(cx + 47, cy - 8), (LABEL_X - 4, row_block)], blk)                  # B sits on the BLOCK row
    arrow(v, [(cx + 36, cy + 3), (162, cy + 3), (162, row_attack), (LABEL_X - 4, row_attack)], att)

    def sub(btn, action):
        lab = button_label(pad_cfg, btn)
        return DESC[action] if len(lab) <= 2 else f"{lab} BUTTON"

    label(v, "move", LABEL_X, row_move, move, sub="STICK OR D-PAD")
    label(v, "dash", LABEL_X, row_dash, dash, sub=sub("y", "dash"))
    label(v, "block", LABEL_X, row_block, blk, sub=sub("b", "block"))
    label(v, "attack", LABEL_X, row_attack, att, sub=sub("a", "attack"))
    v.text(f"{button_label(pad_cfg, 'start')} = PAUSE", cx, cy + 46, v.c["dim"], center=True)
    if note:
        v.text(note, v.W // 2, 177, v.c["act_move"], center=True)
    else:
        pad_indicator(v, 24, 176, connected, name, generic)
