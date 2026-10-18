"""UI state machine for the game shell. Pure logic: no pygame, no drawing, no wall clock.

Input arrives as command strings (up/down/left/right/confirm/back/help/scanlines/quit), time as `update(dt, poll)`.
Only a MatchSession steps the game, and only in the PLAY state, so menus, countdown, pause and results can never
change game state. The view reads these attributes and the session's game; it writes nothing back.
"""
from adaptive_npc.npc.live import MissingCheckpointError
from adaptive_npc.render.clock import TickClock

TITLE, CONTROLS, SELECT, COUNTDOWN, PLAY, ROUND_END, PAUSE, RESULTS, QUIT = (
    "title", "controls", "select", "countdown", "play", "round_end", "pause", "results", "quit")
IN_MATCH = (COUNTDOWN, PLAY, ROUND_END, PAUSE)

TITLE_ITEMS = ("PLAY", "CONTROLS", "QUIT")
PAUSE_ITEMS = ("RESUME", "RESTART MATCH", "QUIT TO MENU")


class Shell:
    def __init__(self, ui_cfg, enemies, tick_rate, session_factory, has_menu=True, select_enabled=True,
                 enemy_index=0, blind=False, label="A", on_match_end=None, input_clear=None, max_catchup=4, panel=False):
        self.cfg, self.enemies = ui_cfg, list(enemies)
        self.blind, self.label = bool(blind), label
        self.has_menu, self.select_enabled = bool(has_menu), bool(select_enabled)
        self.make_session, self.on_match_end = session_factory, on_match_end
        self.input_clear = input_clear or (lambda: None)
        self.max_catchup = int(max_catchup)
        self.speeds = sorted({*(float(s) for s in ui_cfg["speed"]["options"]), float(tick_rate)})
        self.speed_idx = self.speeds.index(float(tick_rate))
        self.enemy_idx = int(enemy_index)
        self.step_s = float(ui_cfg["countdown_step_seconds"])
        self.legend_seconds = float(ui_cfg["legend_seconds"])
        self.scanlines = bool(ui_cfg["scanlines"])
        self.panel = bool(panel) and not self.blind   # demo panel; never in --blind

        self.t = 0.0                  # wall seconds since start (blink animations only)
        self.state = TITLE if has_menu else QUIT
        self.title_started = False
        self.cursor = 0               # menu cursor in the current menu
        self.error = self.error_text = None   # missing-checkpoint message (lines for the select screen / full text)
        self.session = self.clock = None
        self.count_t = 0.0
        self.pause_from = None
        self.legend_visible, self.legend_timer = False, None
        self.fx = []                  # events of the latest update(), for the view: (kind, player)
        self.results = None
        self._reported = False

    # ------------------------------------------------------------------ queries
    @property
    def enemy(self):
        return self.enemies[self.enemy_idx]

    @property
    def tick_rate(self):
        return self.speeds[self.speed_idx]

    @property
    def opponent_label(self):
        return f"OPPONENT {self.label}" if self.blind else self.enemy["name"]

    @property
    def countdown_text(self):
        """'3' / '2' / '1' / 'FIGHT!' during the countdown, else None."""
        if self.state != COUNTDOWN:
            return None
        k = int(self.count_t // self.step_s)
        return ("3", "2", "1", "FIGHT!")[min(k, 3)]

    @property
    def menu_items(self):
        if self.state == TITLE:
            return TITLE_ITEMS
        if self.state == PAUSE:
            return PAUSE_ITEMS if self.has_menu else PAUSE_ITEMS[:2] + ("QUIT",)
        if self.state == RESULTS:
            return ("REMATCH", "MENU" if self.has_menu else "QUIT")
        return ()

    # ------------------------------------------------------------------ match lifecycle
    def begin(self):
        """Start the match directly (no title / select screens)."""
        self._start_match()

    def _start_match(self):
        try:
            self.session = self.make_session(self.enemy, self.tick_rate)
        except MissingCheckpointError as e:
            self.session, self.error, self.error_text = None, self._error_lines(e), str(e)
            self.state = SELECT if self.select_enabled and self.has_menu else QUIT
            return
        self.error = self.error_text = None
        self.clock = TickClock(self.tick_rate, self.max_catchup)
        self.legend_visible, self.legend_timer = True, self.legend_seconds
        self._reported = False
        self._enter_countdown()

    @staticmethod
    def _error_lines(e):
        lines = ["MISSING MODEL FILE"]
        for path, hint in (e.missing or [(str(e), "")]):
            lines.append(path)
        return lines

    def _enter_countdown(self):
        self.state, self.count_t = COUNTDOWN, 0.0
        self.input_clear()

    def _enter_play(self):
        self.state = PLAY
        self.clock.reset()
        self.input_clear()

    def _report(self, completed):
        """Hand the match summary to on_match_end once (only if at least one round was played)."""
        if self.session is None or self._reported or not self.session.rounds:
            return
        self._reported = True
        if self.on_match_end:
            self.on_match_end(self.session.summary(completed))

    def _leave_match(self):
        """Quit to menu (or quit if there is none)."""
        s = self.session
        if s is not None and s.n_rounds == 0 and s.rounds:  # endless match: quitting shows its results
            self._finish()
            return
        self._report(False)
        self.session = None
        self.cursor = 0
        if self.has_menu:
            self.state, self.title_started = TITLE, True
        else:
            self.state = QUIT

    def _finish(self):
        self._report(True)
        self.state, self.cursor = RESULTS, 0

    # ------------------------------------------------------------------ input
    def handle(self, cmd):
        if cmd == "quit":
            self._report(False)
            self.state = QUIT
        elif cmd == "scanlines":
            self.scanlines = not self.scanlines
        elif cmd == "panel":
            self.panel = not self.panel and not self.blind
        elif cmd == "help":
            self.legend_visible, self.legend_timer = not self.legend_visible, None
        else:
            getattr(self, "_on_" + self.state)(cmd)

    def _move(self, cmd):
        if cmd in ("up", "down"):
            self.cursor = (self.cursor + (1 if cmd == "down" else -1)) % len(self.menu_items)

    def _on_title(self, cmd):
        if not self.title_started:
            if cmd == "confirm":
                self.title_started, self.cursor = True, 0
            return
        self._move(cmd)
        if cmd == "back":
            self.title_started = False
        elif cmd == "confirm":
            item = TITLE_ITEMS[self.cursor]
            if item == "PLAY":
                if self.select_enabled:
                    self.state, self.error = SELECT, None
                else:
                    self._start_match()
            elif item == "CONTROLS":
                self.state = CONTROLS
            else:
                self.state = QUIT

    def _on_controls(self, cmd):
        if cmd in ("confirm", "back"):
            self.state = TITLE

    def _on_select(self, cmd):
        n = len(self.enemies)
        if cmd in ("up", "down"):
            self.enemy_idx = (self.enemy_idx + (1 if cmd == "down" else -1)) % n
            self.error = None
        elif cmd in ("left", "right"):
            self.speed_idx = min(max(self.speed_idx + (1 if cmd == "right" else -1), 0), len(self.speeds) - 1)
        elif cmd == "back":
            self.state = TITLE
        elif cmd == "confirm":
            self._start_match()

    def _on_countdown(self, cmd):
        if cmd == "back":
            self._pause()

    _on_play = _on_countdown

    def _on_round_end(self, cmd):
        if cmd == "back":
            self._pause()
        elif cmd == "confirm":
            self.session.next_round()
            if self.session.finished:
                self._finish()
            else:
                self._enter_countdown()

    def _pause(self):
        self.pause_from, self.state, self.cursor = self.state, PAUSE, 0

    def _on_pause(self, cmd):
        self._move(cmd)
        if cmd == "back":
            self._resume()
        elif cmd == "confirm":
            item = self.menu_items[self.cursor]
            if item == "RESUME":
                self._resume()
            elif item == "RESTART MATCH":
                self._report(False)
                self._start_match()
            else:
                self._leave_match()

    def _resume(self):
        self.state = self.pause_from
        if self.state == PLAY:
            self.clock.reset()
            self.input_clear()

    def _on_results(self, cmd):
        self._move(cmd)
        if cmd == "confirm":
            if self.cursor == 0:
                self._start_match()
            else:
                self.session = None
                if self.has_menu:
                    self.state, self.title_started = TITLE, True
                else:
                    self.state = QUIT

    # ------------------------------------------------------------------ time
    def update(self, dt, poll):
        """Advance by `dt` wall seconds. `poll()` returns the human's action for one game step. Returns the events
        (kind, player) the steps of this update produced: hit, parry, dodge, ko."""
        self.t += dt
        self.fx = []
        if self.legend_timer is not None and self.state != PAUSE:
            self.legend_timer -= dt
            if self.legend_timer <= 0:
                self.legend_visible, self.legend_timer = False, None
        if self.state == COUNTDOWN:
            self.count_t += dt
            if self.count_t >= self.step_s * 3 + self.step_s * 0.75:
                self._enter_play()
        elif self.state == PLAY:
            for _ in range(self.clock.advance(dt)):
                done = self.session.tick(poll())
                self.fx += self._events()
                if done:
                    self.state = ROUND_END
                    break
        return self.fx

    def _events(self):
        g, out = self.session.game, []
        for i in (0, 1):
            if g.dealt[1 - i] > 0:
                out.append(("hit", i))
            if g.parried[i]:
                out.append(("parry", i))
            if g.invuln[i]:
                out.append(("dodge", i))
            if g.hp[i] <= 0:
                out.append(("ko", i))
        return out
