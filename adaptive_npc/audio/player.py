"""SoundManager: plays the synthesized effects and music on pygame.mixer. Presentation only, like the view.

It reads the shell (state, countdown text, fx events) and never writes to it, so sound cannot change game state, timing
or results. Effects depend only on the event kind, never on which opponent is playing (blind mode leaks nothing).
If the mixer cannot start, or fails later, it prints one warning line and every method becomes a no-op.
"""
import sys

import pygame

from adaptive_npc.audio import synth
from adaptive_npc.ui import shell as S

EVENT_SOUND = (("attack", "attack"), ("block", "block"), ("dodge", "dodge"), ("parry", "parry"),
               ("hit", "hit"), ("ko", "ko"))             # fixed firing order within one frame
MENU_STATES = (S.TITLE, S.CONTROLS, S.BIND, S.SELECT, S.PAUSE, S.RESULTS)
MUSIC_FOR_STATE = {S.TITLE: "menu", S.CONTROLS: "menu", S.BIND: "menu", S.SELECT: "menu", S.COUNTDOWN: "fight", S.PLAY: "fight",
                   S.ROUND_END: "fight", S.PAUSE: "fight"}      # results screen: jingle only
MOVE_CMDS = ("up", "down", "left", "right", "back")


class SoundManager:
    def __init__(self, cfg, mute=False, music=True, record=False):
        """cfg = the `audio:` block of ui.yaml. `record` keeps a log of every fired effect / music change (tests)."""
        self.cfg = cfg
        self.volume = min(max(float(cfg["volume"]), 0.0), 1.0)
        self.muted = bool(mute)
        self.music_on = bool(music)
        self.record, self.log = record, []
        self.enabled = False
        self.sounds, self.tracks, self.music_ch = {}, {}, None
        self.music_name = None
        self._prev = self._count = None
        self._ct = 0.0
        self._warned = False
        self._init_mixer()

    # ------------------------------------------------------------------ device
    def _init_mixer(self):
        try:
            sr = int(self.cfg["sample_rate"])
            pygame.mixer.init(frequency=sr, size=-16, channels=2, buffer=512, allowedchanges=0)
            pygame.mixer.set_num_channels(16)
            pygame.mixer.set_reserved(1)
            self.music_ch = pygame.mixer.Channel(0)
            self.sr = pygame.mixer.get_init()[0]
            seed = int(self.cfg["noise_seed"])
            self.sounds = {k: self._sound(v) for k, v in synth.build_effects(self.sr, seed).items()}
            if self.music_on:
                self.tracks = {k: self._sound(synth.build_music(k, self.sr, seed)) for k in synth.MUSIC}
            self.enabled = True
            self._apply_volume()
        except Exception as e:                       # no device, bad driver, ...: run silently
            self._disable(e)

    def _disable(self, err):
        if not self._warned:
            print(f"warning: audio unavailable ({err}); continuing without sound", file=sys.stderr)
        self._warned = True
        self.enabled = False
        self.sounds, self.tracks, self.music_ch = {}, {}, None

    @staticmethod
    def _sound(x):
        mono = synth.to_int16(x)
        return pygame.mixer.Sound(buffer=mono.repeat(2).tobytes())      # stereo, L = R

    def _apply_volume(self):
        eff = 0.0 if self.muted else self.volume
        mus = eff * float(self.cfg["music_ratio"])
        for s in self.sounds.values():
            s.set_volume(eff)
        for s in self.tracks.values():
            s.set_volume(mus)

    # ------------------------------------------------------------------ playing
    def _fire(self, name):
        if self.record:
            self.log.append(name)
        if self.enabled:
            try:
                self.sounds[name].play()
            except Exception as e:
                self._disable(e)

    def set_music(self, name):
        if name == self.music_name or not self.music_on:
            return
        self.music_name = name
        if self.record:
            self.log.append(f"music:{name}")
        if not self.enabled:
            return
        try:
            if name is None:
                self.music_ch.fadeout(300)
            else:
                self.music_ch.play(self.tracks[name], loops=-1, fade_ms=300)
        except Exception as e:
            self._disable(e)

    # ------------------------------------------------------------------ inputs
    def control(self, cmd):
        """'mute' toggles everything, 'volume_up' / 'volume_down' step the volume (and unmute)."""
        if cmd == "mute":
            self.muted = not self.muted
        else:
            step = float(self.cfg["volume_step"]) * (1 if cmd == "volume_up" else -1)
            self.volume = round(min(max(self.volume + step, 0.0), 1.0), 3)
            self.muted = False
        if self.enabled:
            self._apply_volume()
        if cmd != "mute" or not self.muted:
            self._fire("menu_move")                  # audible feedback at the new level

    def command(self, cmd, state):
        """Menu sounds for a UI command (in play, up/down/... are gameplay keys and stay silent)."""
        if cmd == "confirm" and (state in MENU_STATES or state == S.ROUND_END):
            self._fire("menu_confirm")
        elif cmd in MOVE_CMDS and state in MENU_STATES:
            self._fire("menu_move")

    def observe(self, shell):
        """Call once per frame after shell.update(): state transitions, countdown beeps, game events, music."""
        st, prev = shell.state, self._prev
        if st == S.COUNTDOWN:
            txt = shell.countdown_text
            fresh = prev not in (S.COUNTDOWN, S.PAUSE) or shell.count_t < self._ct      # a new countdown began
            if fresh:
                self._fire("round_start")
                self._count = None
            elif prev == S.PAUSE:                                                       # resumed: no repeat beep
                self._count = txt
            if txt != self._count:
                self._count = txt
                self._fire("fight" if txt == "FIGHT!" else "count")
        self._ct = shell.count_t
        if st == S.RESULTS and prev != S.RESULTS:
            self._fire("win" if shell.session is not None and shell.session.verdict() == "human" else "lose")
        kinds = {k for k, _ in shell.fx}
        for kind, name in EVENT_SOUND:
            if kind in kinds:
                self._fire(name)
        self.set_music(MUSIC_FOR_STATE.get(st))
        self._prev = st
