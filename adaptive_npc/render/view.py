"""Pygame drawing of the arena + HUD. Reads ArenaGame state only; never mutates it."""
import pygame

from adaptive_npc.env.actions import DIRS


class ArenaView:
    def __init__(self, cfg, arena_size, title="Adaptive NPC"):
        self.n = int(arena_size)
        self.tile = int(cfg["tile_size"])
        self.hud_h = int(cfg["hud_height"])
        self.c = {k: tuple(v) for k, v in cfg["colors"].items()}
        self.w = self.n * self.tile
        self.h = self.hud_h + self.n * self.tile
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas,dejavusansmono,arial", max(14, self.tile // 4))
        self.big = pygame.font.SysFont("consolas,dejavusansmono,arial", max(24, self.tile // 2), bold=True)

    # -------------------------------------------------------------- primitives
    def _text(self, s, pos, font=None, color=None, center=False):
        surf = (font or self.font).render(s, True, color or self.c["text"])
        rect = surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surf, rect)

    def _bar(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.screen, self.c["grid"], (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, int(w * max(0.0, min(1.0, frac))), h))

    def _cell(self, gx, gy):
        return pygame.Rect(gx * self.tile, self.hud_h + gy * self.tile, self.tile, self.tile)

    # -------------------------------------------------------------- drawing
    def _draw_fighter(self, g, i, color):
        r = self._cell(g.x[i], g.y[i]).inflate(-self.tile // 4, -self.tile // 4)
        body = tuple(255 - (255 - v) // 3 for v in color) if g.invuln[i] else color
        pygame.draw.rect(self.screen, body, r, border_radius=self.tile // 8)
        if g.stun[i] > 0:
            pygame.draw.rect(self.screen, self.c["stun"], r, width=4, border_radius=self.tile // 8)
        if g.dealt[1 - i] > 0:
            pygame.draw.rect(self.screen, self.c["hit"], r, width=4, border_radius=self.tile // 8)
        dx, dy = DIRS[g.face[i]]
        cx, cy = r.center
        reach = self.tile // 2
        pygame.draw.line(self.screen, self.c["text"], (cx, cy), (cx + dx * reach, cy + dy * reach), 4)
        if g.blocking[i]:  # shield bar across the front
            px, py = -dy, dx
            mx, my = cx + dx * (reach - 2), cy + dy * (reach - 2)
            half = self.tile // 3
            pygame.draw.line(self.screen, self.c["block"],
                             (mx + px * half, my + py * half), (mx - px * half, my - py * half), 8)

    def _draw_hud(self, g, score, round_idx, names):
        pygame.draw.rect(self.screen, self.c["hud"], (0, 0, self.w, self.hud_h))
        half = self.w // 2
        for i, (x0, col) in enumerate(((8, self.c["npc"]), (half + 8, self.c["player"]))):
            bw = half - 24
            self._text(names[i], (x0, 4), color=col)
            self._text(f"HP {g.hp[i]}", (x0 + bw - 56, 4))
            self._bar(x0, 28, bw, 12, g.hp[i] / g.hp_max, self.c["hp"])
            self._bar(x0, 46, bw, 8, g.sta[i] / g.sta_max, self.c["stamina"])
        self._text(f"R{round_idx + 1}  t={g.t}/{g.max_steps}  opp-you-draw {score[0]}-{score[1]}-{score[2]}",
                   (8, self.hud_h - 18))

    def draw(self, g, score=(0, 0, 0), round_idx=0, names=("NPC", "YOU"), banner=None):
        """Render one frame. `score` = (opponent wins, human wins, draws)."""
        self.screen.fill(self.c["background"])
        for gx in range(self.n):
            for gy in range(self.n):
                pygame.draw.rect(self.screen, self.c["grid"], self._cell(gx, gy), 1)
        self._draw_fighter(g, 0, self.c["npc"])
        self._draw_fighter(g, 1, self.c["player"])
        self._draw_hud(g, score, round_idx, names)
        if banner:
            shade = pygame.Surface((self.w, self.h - self.hud_h), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 150))
            self.screen.blit(shade, (0, self.hud_h))
            lines = banner if isinstance(banner, (list, tuple)) else [banner]
            for k, line in enumerate(lines):
                self._text(line, (self.w // 2, self.hud_h + self.h // 3 + k * self.tile // 2),
                           font=self.big if k == 0 else self.font, center=True)
        pygame.display.flip()
