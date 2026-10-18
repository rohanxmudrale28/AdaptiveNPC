"""Retro 8-bit view: the whole game is drawn on a small internal surface (default 320x240, 16-colour palette from
configs/ui.yaml) and scaled to the window with nearest-neighbour. Reads Shell / MatchSession / ArenaGame state only;
it never mutates the game. Animation timers live here and are purely visual."""
import random

import pygame

from adaptive_npc.env.actions import DIRS
from adaptive_npc.render import font
from adaptive_npc.render.sprites import SIZE, SpriteSet
from adaptive_npc.ui import shell as S

KEY_LABELS = {"left shift": "LSHIFT", "right shift": "RSHIFT", "left ctrl": "LCTRL", "right ctrl": "RCTRL",
              "return": "ENTER", "escape": "ESC", "space": "SPACE", "up": "UP", "down": "DOWN", "left": "LEFT",
              "right": "RIGHT"}
HUD_H = 46
BAR_W = 96
STAR = ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))


def key_label(name):
    return KEY_LABELS.get(name, name.upper())


def keys_text(names, sep=" / "):
    return sep.join(key_label(k) for k in names)


class RetroView:
    def __init__(self, ui_cfg, render_cfg, arena_size, caption="ADAPTIVE NPC"):
        self.cfg, self.keys_cfg = ui_cfg, render_cfg["keys"]
        self.pal = {k: tuple(v) for k, v in ui_cfg["palette"].items()}
        if len(self.pal) > 16:
            raise ValueError(f"palette has {len(self.pal)} colours, at most 16 allowed")
        self.c = {role: self.pal[name] for role, name in ui_cfg["colors"].items()}
        self.tag_color = {t: self.pal[n] for t, n in ui_cfg["tags"].items()}
        self.W, self.H = (int(v) for v in ui_cfg["internal"])
        self.scale = int(ui_cfg["window_scale"])
        self.n = int(arena_size)
        self.ax, self.ay = (self.W - self.n * SIZE) // 2, HUD_H + 6
        self.window = pygame.display.set_mode((self.W * self.scale, self.H * self.scale))
        pygame.display.set_caption(caption)
        self.canvas = pygame.Surface((self.W, self.H))
        self.knight = SpriteSet("knight", self.pal, ui_cfg["looks"]["player"])
        self._robots = {}
        self._scan = None
        self.rng = random.Random(0)  # cosmetic shake only
        self.time, self.shake_t, self.shake_amp = 0.0, 0.0, 0
        self.shake_cfg = ui_cfg["shake"]
        self.reset_fx()

    # ------------------------------------------------------------------ small helpers
    def reset_fx(self):
        self.last_t = -1
        self.cur = [None, None]
        self.prev = [None, None]
        self.t_move = [-9.0, -9.0]
        self.walk_n = [0, 0]
        self.facing_ko = [None, None]

    def robot(self, look, accessory):
        key = (look, accessory)
        if key not in self._robots:
            self._robots[key] = SpriteSet("robot", self.pal, self.cfg["looks"][look], accessory)
        return self._robots[key]

    def text(self, s, x, y, color=None, scale=1, **kw):
        return font.draw_text(self.canvas, s, x, y, color or self.c["text"], scale, **kw)

    def box(self, x, y, w, h, fill=None, edge=None):
        if fill is not None:
            pygame.draw.rect(self.canvas, fill, (x, y, w, h))
        if edge is not None:
            pygame.draw.rect(self.canvas, edge, (x, y, w, h), 1)

    def blink(self, hz=2.0):
        return int(self.time * hz * 2) % 2 == 0

    def enemy_sprites(self, shell):
        if shell.blind:
            return self.robot("npc", "none")
        e = shell.enemy
        return self.robot(e["look"], e["accessory"])

    def blit_sprite(self, surf, x, y, scale=1):
        if scale != 1:
            surf = pygame.transform.scale(surf, (SIZE * scale, SIZE * scale))
        self.canvas.blit(surf, (int(x), int(y)))

    # ------------------------------------------------------------------ frame
    def draw(self, shell, dt=0.0):
        self.time += dt
        for kind, who in shell.fx:
            if kind == "hit":
                self.kick(self.shake_cfg["hit"])
            elif kind == "parry":
                self.kick(self.shake_cfg["parry"])
        in_match = shell.session is not None and shell.state in (S.COUNTDOWN, S.PLAY, S.ROUND_END, S.PAUSE)
        if in_match:
            self.sync(shell.session.game)
        self.canvas.fill(self.pal["ink"])
        {S.TITLE: self.draw_title, S.CONTROLS: self.draw_controls, S.SELECT: self.draw_select,
         S.COUNTDOWN: self.draw_match, S.PLAY: self.draw_match, S.ROUND_END: self.draw_match,
         S.PAUSE: self.draw_match, S.RESULTS: self.draw_results, S.QUIT: lambda s: None}[shell.state](shell)
        if shell.state == S.PAUSE:
            self.draw_pause(shell)
        self.present(shell.scanlines, dt)

    def kick(self, amp):
        self.shake_t, self.shake_amp = float(self.shake_cfg["seconds"]), int(amp)

    def present(self, scanlines, dt):
        ox = oy = 0
        if self.shake_t > 0:
            self.shake_t -= dt
            k = self.shake_amp * max(self.shake_t, 0) / float(self.shake_cfg["seconds"])
            a = int(round(k))
            if a:
                ox, oy = self.rng.randint(-a, a), self.rng.randint(-a, a)
        sc = self.scale
        if ox or oy:
            self.window.fill(self.pal["ink"])
            frame = pygame.transform.scale(self.canvas, (self.W * sc, self.H * sc))
            self.window.blit(frame, (ox * sc, oy * sc))
        else:
            pygame.transform.scale(self.canvas, (self.W * sc, self.H * sc), self.window)
        if scanlines:
            self.window.blit(self.scan_overlay(), (0, 0))
        pygame.display.flip()

    def scan_overlay(self):
        if self._scan is None:
            sc = self.scale
            s = pygame.Surface((self.W * sc, self.H * sc), pygame.SRCALPHA)
            for y in range(sc - 1, self.H * sc, sc):
                pygame.draw.line(s, (0, 0, 0, 70), (0, y), (self.W * sc, y))
            self._scan = s
        return self._scan

    # ------------------------------------------------------------------ match
    def sync(self, g):
        """Track fighter positions per game tick (visual tweening only)."""
        if g.t < self.last_t or self.cur[0] is None:  # new round
            self.cur = [(g.x[0], g.y[0]), (g.x[1], g.y[1])]
            self.prev = list(self.cur)
            self.walk_n = [0, 0]
            self.t_move = [-9.0, -9.0]
        elif g.t != self.last_t:
            for i in (0, 1):
                self.prev[i] = self.cur[i]
                self.cur[i] = (g.x[i], g.y[i])
                self.t_move[i] = self.time
                if self.cur[i] != self.prev[i]:
                    self.walk_n[i] += 1
        self.last_t = g.t

    def tile_xy(self, gx, gy):
        return self.ax + gx * SIZE, self.ay + gy * SIZE

    def draw_match(self, shell):
        s = shell.session
        g = s.game
        self.draw_arena(g)
        self.draw_fighters(shell, g)
        self.draw_hud(shell, g)
        self.draw_strip(shell)
        if shell.state == S.COUNTDOWN:
            self.text(shell.countdown_text, self.W // 2, self.ay + self.n * SIZE // 2 - 17, self.c["title"], 5,
                      center=True, shadow=self.c["title_shadow"])

    def draw_arena(self, g):
        for gx in range(self.n):
            for gy in range(self.n):
                x, y = self.tile_xy(gx, gy)
                self.box(x, y, SIZE, SIZE, self.c["floor_a"] if (gx + gy) % 2 == 0 else self.c["floor_b"])
        w = self.n * SIZE
        pygame.draw.rect(self.canvas, self.c["wall"], (self.ax - 2, self.ay - 2, w + 4, w + 4), 2)

    def pos_of(self, i, shell):
        """Visual position (pixels) with a short tween from the previous tile."""
        dur = min(0.12, 0.5 / shell.tick_rate)
        p = max(0.0, min(1.0, (self.time - self.t_move[i]) / dur))
        (px, py), (cx, cy) = self.prev[i], self.cur[i]
        return self.ax + (px + (cx - px) * p) * SIZE, self.ay + (py + (cy - py) * p) * SIZE

    def sprites_of(self, i, shell):
        return self.knight if i == 1 else self.enemy_sprites(shell)

    def draw_fighters(self, shell, g):
        order = sorted((0, 1), key=lambda i: self.pos_of(i, shell)[1])
        for i in order:
            self.draw_fighter(shell, g, i)
        for i in (0, 1):
            self.draw_effects(shell, g, i)

    def draw_fighter(self, shell, g, i):
        sp = self.sprites_of(i, shell)
        face = g.face[i]
        age = self.time - self.t_move[i]
        x, y = self.pos_of(i, shell)
        if g.hp[i] <= 0:
            if self.blink(4):
                self.blit_sprite(sp.frames[face]["ko"], x, y)
            return
        attacking = g.cd[i] == g.atk_cooldown and g.stun[i] == 0 and not g.blocking[i]
        if g.blocking[i]:
            name = "block"
        elif attacking:
            name = "attack"
        elif self.cur[i] != self.prev[i] and not g.invuln[i]:
            name = f"walk{self.walk_n[i] % 2}"
        else:
            name = "idle1" if int((self.time + 0.5 * i) * 2) % 2 else "idle0"
        if attacking:  # lunge toward the target, settling back
            dx, dy = DIRS[face]
            k = round(3 * max(0.0, 1 - age / 0.2))
            x, y = x + dx * k, y + dy * k
        if g.invuln[i]:
            self.draw_trail(shell, g, i, sp)
            if int(self.time * 20) % 2:  # i-frames: blink
                return
        frame = sp.frames[face][name]
        if g.dealt[1 - i] > 0 and age < 0.25 and int(age * 24) % 2 == 0:  # hit flash
            frame = sp.silhouette(frame, self.c["hit"])
        self.blit_sprite(frame, x, y)

    def draw_trail(self, shell, g, i, sp):
        (px, py), (cx, cy) = self.prev[i], self.cur[i]
        if (px, py) == (cx, cy):
            return
        ghost = sp.frames[g.face[i]]["idle0"]
        for k, alpha in ((0.25, 80), (0.5, 120), (0.75, 170)):
            gx, gy = self.ax + (px + (cx - px) * k) * SIZE, self.ay + (py + (cy - py) * k) * SIZE
            img = sp.silhouette(ghost, self.pal["sky"] if i == 1 else self.pal["amber"])
            img.set_alpha(alpha)
            self.canvas.blit(img, (int(gx), int(gy)))

    def draw_effects(self, shell, g, i):
        sp = self.sprites_of(i, shell)
        age = self.time - self.t_move[i]
        x, y = self.pos_of(i, shell)
        face = g.face[i]
        if g.cd[i] == g.atk_cooldown and g.stun[i] == 0 and not g.blocking[i] and g.hp[i] > 0 and age < 0.28:
            dx, dy = DIRS[face]
            arc = sp.arc(face)
            if age > 0.14:
                arc.set_alpha(150)
            self.canvas.blit(arc, (int(x + dx * SIZE), int(y + dy * SIZE)))
        if g.parried[i]:  # the blocker's front tile sparks
            j = 1 - i
            dx, dy = DIRS[g.face[j]]
            bx, by = self.pos_of(j, shell)
            if self.time - self.t_move[j] < 0.3:
                self.draw_star(bx + dx * 10 + 5, by + dy * 10 + 5, self.pal["white"], 2)
        if g.stun[i] > 0 and g.hp[i] > 0:
            for k in range(3):
                a = self.time * 5 + k * 2.094
                sx = x + 7 + 7 * pygame.math.Vector2(1, 0).rotate_rad(a).x
                sy = y - 1 + 2 * pygame.math.Vector2(1, 0).rotate_rad(a).y
                self.draw_star(sx, sy, self.pal["amber"] if k % 2 == 0 else self.c["stun"])

    def draw_star(self, x, y, color, grow=1):
        for dx, dy in STAR:
            pygame.draw.rect(self.canvas, color, (int(x) + (dx - 1) * grow, int(y) + (dy - 1) * grow, grow, grow))

    # ------------------------------------------------------------------ HUD
    def seg_bar(self, x, y, w, h, value, vmax, color, right_to_left=False):
        gap = 1
        seg = max(1, (w - gap * (vmax - 1)) // vmax)
        for k in range(vmax):
            kk = vmax - 1 - k if right_to_left else k
            sx = x + k * (seg + gap)
            self.box(sx, y, seg, h, color if kk < value else self.c["bar_back"])

    def draw_hud(self, shell, g):
        s = shell.session
        npc_col = self.pal[self.cfg["looks"]["npc" if shell.blind else shell.enemy["look"]]["a"]]
        ply_col = self.pal[self.cfg["looks"]["player"]["a"]]
        self.text(shell.opponent_label, 6, 4, npc_col)
        self.text("YOU", self.W - 6, 4, ply_col, right=True)
        bar_w = BAR_W - 14
        bx_l, bx_r = 6 + 14, self.W - 6 - bar_w
        for x, label_x, rtl, who in ((bx_l, 6, False, 0), (bx_r, bx_r - 14, True, 1)):
            self.text("HP", label_x, 16, self.c["dim"])
            self.seg_bar(x, 15, bar_w, 8, g.hp[who], g.hp_max, self.c["hp"], right_to_left=rtl)
            self.text("ST", label_x, 28, self.c["dim"])
            self.seg_bar(x, 28, bar_w, 5, g.sta[who], g.sta_max, self.c["stamina"], right_to_left=rtl)
        rnd = f"ROUND {s.round + 1}" + (f"/{s.n_rounds}" if s.n_rounds else "")
        self.text(rnd, self.W // 2, 4, self.c["title"], center=True)
        self.text(str(s.score[0]), self.W // 2 - 14, 15, npc_col, 2, right=True)
        self.text("-", self.W // 2, 15, self.c["dim"], 2, center=True)
        self.text(str(s.score[1]), self.W // 2 + 14, 15, ply_col, 2)
        if s.score[2]:
            self.text(f"DRAWS {s.score[2]}", self.W // 2, 33, self.c["dim"], center=True)
        frac = min(1.0, g.t / g.max_steps)
        self.box(0, HUD_H - 2, self.W, 2, self.c["bar_back"])
        self.box(0, HUD_H - 2, int(self.W * frac), 2, self.c["dim"])

    def legend_lines(self):
        k = self.keys_cfg
        move = "".join(k[d][0].upper() for d in ("up", "left", "down", "right"))
        if len(move) != 4:
            move = "/".join(k[d][0].upper() for d in ("up", "left", "down", "right"))
        return [f"MOVE {move}   ATTACK {key_label(k['attack'][0])}   BLOCK {key_label(k['block'][0])}"
                f"   DODGE {key_label(k['dodge'][0])}",
                f"H LEGEND   F SCANLINES   ESC PAUSE"]

    def draw_strip(self, shell):
        y0 = self.ay + self.n * SIZE + 8
        s = shell.session
        if shell.state == S.ROUND_END:
            g = s.game
            msg = {1: "YOU WIN THE ROUND!", 0: f"{shell.opponent_label} WINS THE ROUND", -1: "DRAW"}[g.winner]
            self.text(msg, self.W // 2, y0, self.c["title"], center=True, shadow=self.c["title_shadow"])
            if self.blink(1.5):
                last = s.n_rounds != 0 and s.round + 1 >= s.n_rounds
                self.text("PRESS ENTER " + ("FOR RESULTS" if last else "FOR NEXT ROUND"), self.W // 2, y0 + 12,
                          self.c["text"], center=True)
        elif shell.legend_visible:
            for k, line in enumerate(self.legend_lines()):
                self.text(line, self.W // 2, y0 + k * 12, self.c["text"] if k == 0 else self.c["dim"], center=True)
        else:
            self.text("H FOR CONTROLS", self.W // 2, y0 + 12, self.c["dim"], center=True)

    # ------------------------------------------------------------------ pause
    def draw_pause(self, shell):
        shade = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.canvas.blit(shade, (0, 0))
        items = shell.menu_items
        h = 30 + len(items) * 14
        x, y = self.W // 2 - 60, self.H // 2 - h // 2
        self.box(x, y, 120, h, self.c["floor_a"], self.c["text"])
        self.text("PAUSED", self.W // 2, y + 8, self.c["title"], center=True)
        self.menu(items, shell.cursor, self.W // 2, y + 24, 14)

    def menu(self, items, cursor, cx, y, step, colors=None):
        for k, item in enumerate(items):
            on = k == cursor
            col = self.c["text"] if on else self.c["dim"]
            w = font.text_width(item)
            self.text(item, cx, y + k * step, col, center=True)
            if on and self.blink(3):
                self.text(">", cx - w // 2 - 10, y + k * step, self.c["title"])

    # ------------------------------------------------------------------ screens
    def draw_title(self, shell):
        cfg = self.cfg
        for k in range(0, self.W, 16):  # pixel border
            self.box(k, 0, 8, 4, self.c["wall"])
            self.box(k + 8, self.H - 4, 8, 4, self.c["wall"])
        self.text(cfg["title"], self.W // 2, 34, self.c["title"], 3, center=True, shadow=self.c["title_shadow"])
        self.text(cfg["subtitle"], self.W // 2, 66, self.c["text"], center=True)
        idle = "idle1" if int(self.time * 2) % 2 else "idle0"
        self.blit_sprite(self.knight.frames[3][idle], 52, 96, 3)
        self.blit_sprite(self.robot("npc", "none").frames[2][idle], self.W - 52 - 48, 96, 3)
        if not shell.title_started:
            if self.blink(1.5):
                self.text("PRESS START", self.W // 2, 150, self.c["text"], 2, center=True)
        else:
            self.menu(S.TITLE_ITEMS, shell.cursor, self.W // 2, 134, 16)
        self.text("ENTER SELECT   ARROWS MOVE", self.W // 2, self.H - 16, self.c["dim"], center=True)

    def draw_controls(self, shell):
        k = self.keys_cfg
        self.text("CONTROLS", self.W // 2, 12, self.c["title"], 2, center=True, shadow=self.c["title_shadow"])
        rows = [("MOVE", " ".join(k["up"][:1] + k["left"][:1] + k["down"][:1] + k["right"][:1]).upper()
                 + "  OR ARROWS", "MOVING SETS WHICH WAY YOU FACE"),
                ("ATTACK", keys_text(k["attack"], " OR "), "HITS THE TILE YOU FACE"),
                ("BLOCK", keys_text(k["block"], " OR "), "HOLD: STOPS A FRONT HIT, STUNS ATTACKER"),
                ("DODGE", keys_text(k["dodge"], " OR "), "DASH TWO TILES, CANNOT BE HIT")]
        y = 44
        for name, keys, desc in rows:
            self.text(name, 16, y, self.c["title"])
            self.text(keys, 70, y, self.c["text"])
            self.text(desc, 70, y + 10, self.c["dim"])
            y += 28
        self.text("ATTACK BEATS MOVE   BLOCK BEATS ATTACK", self.W // 2, 160, self.c["text"], center=True)
        self.text("FLANKING BEATS BLOCK   BLOCK AND DODGE COST STAMINA", self.W // 2, 172, self.c["dim"], center=True)
        self.text(f"{key_label(k['quit'][0])} PAUSE   H LEGEND   F SCANLINES", self.W // 2, 192, self.c["dim"],
                  center=True)
        if self.blink(1.5):
            self.text("PRESS ENTER", self.W // 2, self.H - 22, self.c["text"], center=True)

    def draw_select(self, shell):
        self.text("CHOOSE OPPONENT", self.W // 2, 10, self.c["title"], 2, center=True, shadow=self.c["title_shadow"])
        for k, e in enumerate(shell.enemies):
            y = 36 + k * 20
            on = k == shell.enemy_idx
            col = self.pal[self.cfg["looks"][e["look"]]["a"]]
            if on:
                self.box(8, y - 3, 196, 16, self.c["floor_a"], self.c["wall"])
                if self.blink(3):
                    self.text(">", 12, y + 1, self.c["title"])
            self.text(e["name"], 24, y + 1, col if on else self.c["dim"])
            self.text(e["tag"], 196, y + 1, self.tag_color[e["tag"]], right=True)
        e = shell.enemy
        sp = self.robot(e["look"], e["accessory"])
        idle = "idle1" if int(self.time * 2) % 2 else "idle0"
        self.box(222, 38, 80, 68, self.c["floor_a"], self.c["wall"])
        self.blit_sprite(sp.frames[1][idle], 238, 46, 3)
        self.text(e["name"], 262, 96, self.c["text"], center=True)
        self.text(e["desc"], self.W // 2, 142, self.c["text"], center=True)
        self.text("SPEED", 24, 164, self.c["dim"])
        rate = shell.tick_rate
        txt = f"{rate:g} STEPS/S"
        self.text("<", 70, 164, self.c["title"] if shell.speed_idx > 0 else self.c["wall"])
        self.text(txt, 130, 164, self.c["text"], center=True)
        self.text(">", 184, 164, self.c["title"] if shell.speed_idx < len(shell.speeds) - 1 else self.c["wall"])
        if shell.error:
            self.text(shell.error[0], self.W // 2, 184, self.c["hit"], center=True)
            for k2, line in enumerate(shell.error[1:3]):
                self.text(line[-50:], self.W // 2, 196 + k2 * 10, self.c["hit"], center=True)
        self.text("ENTER FIGHT   ESC BACK   LEFT/RIGHT SPEED", self.W // 2, self.H - 14, self.c["dim"], center=True)

    def draw_results(self, shell):
        s = shell.session
        w = s.verdict()
        title = {"human": "YOU WIN!", "npc": f"{shell.opponent_label} WINS", "draw": "DRAW"}[w]
        self.text("MATCH OVER", self.W // 2, 14, self.c["text"], 2, center=True)
        self.text(title, self.W // 2, 36, self.c["title"], 2, center=True, shadow=self.c["title_shadow"])
        npc_col = self.pal[self.cfg["looks"]["npc" if shell.blind else shell.enemy["look"]]["a"]]
        rows = [("ROUNDS WON", f"YOU {s.score[1]}   {shell.opponent_label} {s.score[0]}   DRAWS {s.score[2]}"),
                ("DAMAGE DEALT", str(s.dmg_dealt)), ("DAMAGE TAKEN", str(s.dmg_taken))]
        if not shell.blind:
            acc = s.pred_accuracy
            rows.append(("NPC PREDICTION", "N/A (NO PREDICTOR)" if acc is None else f"{acc * 100:.0f}% OF YOUR MOVES"))
        y = 66
        for label, val in rows:
            self.text(label, 24, y, self.c["dim"])
            self.text(val, 130, y, self.c["text"])
            y += 16
        self.blit_sprite(self.knight.frames[1]["idle0"], 40, 150, 2)
        self.blit_sprite(self.enemy_sprites(shell).frames[1]["idle0"], self.W - 72, 150, 2)
        self.menu(shell.menu_items, shell.cursor, self.W // 2, 160, 16)
