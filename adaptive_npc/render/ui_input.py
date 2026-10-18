"""pygame events -> shell commands (up/down/left/right/confirm/back/help/scanlines/quit)."""
import pygame


class UIKeys:
    def __init__(self, ui_keys, render_keys):
        self.bind = {}
        for cmd, names in (*((k, ui_keys[k]) for k in ("confirm", "back", "help", "scanlines", "panel", "screenshot")),
                           *((k, render_keys[k]) for k in ("up", "down", "left", "right"))):
            self.bind[cmd] = {pygame.key.key_code(n) for n in names}

    def command(self, event):
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            for cmd, keys in self.bind.items():
                if event.key in keys:
                    return cmd
        return None
