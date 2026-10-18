"""Sound (M12 stage 3): synthesized buffers, seamless loops, silent fallback without a device, sound cannot change
game state / frames / results, and blind mode plays the same effects for every opponent. Dummy SDL drivers, so it
runs on any machine."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from adaptive_npc.audio import synth
from adaptive_npc.audio.player import SoundManager
from adaptive_npc.config import load_config
from adaptive_npc.ui import shell as S
from tests.frame_hash import fixed_frames
from tests.test_shell import TICK, cfgs, make_shell, snapshot  # noqa: F401  (cfgs is a fixture)

SR = 22050
AUDIO = load_config("configs/ui.yaml")["audio"]


@pytest.fixture(autouse=True)
def _mixer_closed():
    yield
    pygame.mixer.quit()


# ---------------------------------------------------------------------------------------------- buffers
def test_synth_does_not_import_pygame():
    src = open(synth.__file__, encoding="utf-8").read()
    assert "pygame" not in src.replace("no pygame", "")


@pytest.mark.parametrize("name", list(synth.EFFECT_SECONDS))
def test_effect_buffers_valid(name):
    x = synth.build_effects(SR, 0)[name]
    assert x.shape == (round(synth.EFFECT_SECONDS[name] * SR),)
    assert np.isfinite(x).all() and np.abs(x).max() > 0.1             # non-empty, audible
    assert np.abs(x).max() <= synth.EFFECT_PEAK + 1e-9                # no clipping
    assert x[0] == 0.0 and x[-1] == 0.0                               # no click at either end
    i16 = synth.to_int16(x)
    assert i16.dtype == np.int16 and np.abs(i16.astype(int)).max() < 32767


def test_effects_are_deterministic_and_distinct():
    a, b = synth.build_effects(SR, 0), synth.build_effects(SR, 0)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(synth.build_effects(SR, 1)["attack"], a["attack"])   # noise follows the seed
    assert len({a[k].tobytes() for k in a}) == len(a)


@pytest.mark.parametrize("name", list(synth.MUSIC))
def test_music_buffers_valid_and_loop_seamlessly(name):
    x = synth.build_music(name, SR, 0)
    spec = synth.MUSIC[name]
    steps = len(next(v[-1] for k, v in spec.items() if k != "bpm"))
    step_n = round(60.0 / spec["bpm"] / 2.0 * SR)
    assert x.shape == (steps * step_n,) and 5 < len(x) / SR < 20       # whole steps: the loop is sample-exact
    assert np.isfinite(x).all() and 0.3 < np.abs(x).max() <= synth.MUSIC_PEAK + 1e-9
    assert x[0] == 0.0 and x[-1] == 0.0
    looped = np.concatenate([x[-200:], x[:200]])                       # what the player hears at the join
    seam = np.abs(np.diff(looped))[199]
    assert seam <= np.abs(np.diff(x)).max()                            # the join is no bigger a step than inside the tune
    assert np.abs(x[:20]).max() < 0.2 and np.abs(x[-20:]).max() < 0.2  # and it is near silence on both sides


def test_music_is_quieter_than_effects():
    fx = synth.build_effects(SR, 0)
    rms = lambda v: np.sqrt(np.mean(v ** 2))
    assert max(rms(synth.build_music(n, SR, 0)) for n in synth.MUSIC) * AUDIO["music_ratio"] < \
        0.5 * min(rms(fx[k]) for k in ("hit", "attack", "menu_confirm"))


# ---------------------------------------------------------------------------------------------- the mixer
def test_manager_plays_and_controls_volume():
    sm = SoundManager(AUDIO)
    assert sm.enabled and pygame.mixer.get_init() is not None
    assert sm.volume == 0.5
    sm.control("volume_up")
    assert sm.volume == pytest.approx(0.6) and sm.sounds["hit"].get_volume() == pytest.approx(0.6, abs=0.01)
    for _ in range(20):
        sm.control("volume_down")
    assert sm.volume == 0.0
    sm.control("volume_up")
    sm.control("mute")
    assert sm.muted and sm.sounds["hit"].get_volume() == 0.0 and sm.tracks["fight"].get_volume() == 0.0
    sm.control("mute")
    assert not sm.muted and sm.sounds["hit"].get_volume() > 0.0
    assert sm.tracks["fight"].get_volume() == pytest.approx(sm.volume * AUDIO["music_ratio"], abs=0.01)


def test_no_music_builds_no_tracks():
    sm = SoundManager(AUDIO, music=False, record=True)
    sm.set_music("fight")
    assert sm.tracks == {} and sm.log == []


def test_no_audio_device_is_silent_not_a_crash(monkeypatch, capsys):
    def boom(*a, **k):
        raise pygame.error("No audio device found")
    monkeypatch.setattr(pygame.mixer, "init", boom)
    sm = SoundManager(AUDIO, record=True)
    assert not sm.enabled
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("warning: audio unavailable")
    sm.control("volume_up"), sm.control("mute"), sm.command("confirm", S.TITLE), sm.set_music("menu")   # all no-ops
    assert capsys.readouterr().err == ""


def test_play_main_starts_without_a_mixer(monkeypatch, capsys):
    from scripts import play
    monkeypatch.setattr(pygame.mixer, "init", lambda *a, **k: (_ for _ in ()).throw(pygame.error("no device")))
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    play.main(["--opponent", "dummy", "--rounds", "1"])
    assert "warning: audio unavailable" in capsys.readouterr().err


def test_play_flags():
    from scripts import play
    a = play.build_parser().parse_args(["--mute", "--no-music"])
    assert a.mute and a.no_music
    with pytest.raises(SystemExit):
        play.build_parser().parse_args(["--music", "--no-music"])


# ---------------------------------------------------------------------------------------------- events
def drive(sh, sm, human, steps, listen=True):
    """Advance `steps` game steps with scripted human actions, as play.py's loop would, optionally with sound."""
    it = iter(human)
    for _ in range(steps):
        sh.update(1.0 / TICK, lambda: next(it, 0))
        if listen:
            sm.observe(sh)
        if sh.state == S.ROUND_END:
            break


def start_match(cfgs, enemy="dummy", **kw):
    sh, ended, env = make_shell(cfgs, rounds=1, env_over=["random_start=false"], **kw)
    sh.enemy_idx = next(i for i, e in enumerate(sh.enemies) if e["key"] == enemy)
    sh.handle("confirm"), sh.handle("confirm"), sh.handle("confirm")
    return sh, ended


def test_countdown_and_game_events_fire_sounds(cfgs):
    sh, _ = start_match(cfgs)
    sm = SoundManager(AUDIO, music=False, record=True)
    while sh.state == S.COUNTDOWN:
        sh.update(0.1, lambda: 0)
        sm.observe(sh)
    assert sm.log == ["round_start", "count", "count", "count", "fight"]
    # kinds from game state: attack (5), block (6), dodge (7) by the human
    sm.log.clear()
    sh.fx = [("attack", 1), ("hit", 0), ("ko", 0)]
    sm.observe(sh)
    assert sm.log == ["attack", "hit", "ko"]
    g = sh.session.game
    sm.log.clear()
    sh.update(1.0 / TICK, lambda: 6)          # human blocks
    sm.observe(sh)
    assert "block" in sm.log and g.blocking[1]
    sm.log.clear()
    sh.update(1.0 / TICK, lambda: 6)          # still holding block: no second clang
    sm.observe(sh)
    assert "block" not in sm.log
    sm.log.clear()
    sh.update(1.0 / TICK, lambda: 7)
    sm.observe(sh)
    assert "dodge" in sm.log


def test_menu_sounds_results_and_music(cfgs):
    sh = make_shell(cfgs)[0]
    sm = SoundManager(AUDIO, record=True)
    sm.observe(sh)
    assert sm.log == ["music:menu"]
    sm.log.clear()
    sm.command("down", sh.state), sm.command("confirm", sh.state)
    sm.command("up", S.PLAY)
    assert sm.log == ["menu_move", "menu_confirm"]
    sm.log.clear()
    sh.state = S.RESULTS
    sh.session = make_shell(cfgs)[0].make_session(sh.enemy, TICK)
    sm.observe(sh)
    assert sm.log == ["lose", "music:None"]           # 0-0: not a human win
    sh.session.score[1] = 1
    sm.log.clear(), setattr(sm, "_prev", S.PLAY)
    sm.observe(sh)
    assert sm.log == ["win"]


# ---------------------------------------------------------------------------------------------- no effect on the game
def play_scripted(cfgs, sound):
    sh, ended = start_match(cfgs, enemy="rusty")
    sm = SoundManager(AUDIO, music=sound, mute=not sound, record=True) if sound is not None else None
    human = [3, 3, 5, 6, 5, 7, 5, 2, 5, 5, 6, 4] * 30
    for _ in range(40):
        drive(sh, sm, human, 1, listen=sm is not None)
        if sh.state == S.ROUND_END:
            break
    return snapshot(sh.session.game), sh.session.score[:], sh.session.summary(True)


def test_sound_does_not_change_game_state_or_results(cfgs):
    quiet, loud = play_scripted(cfgs, None), play_scripted(cfgs, True)
    muted = play_scripted(cfgs, False)
    assert quiet == loud == muted


def test_sound_does_not_change_frames(cfgs):
    from tests.test_shell import make_shell as ms
    def factory(c, **kw):
        return ms(c, **kw)
    base = fixed_frames(cfgs, factory)
    SoundManager(AUDIO, record=True)                  # a live manager in the same process
    assert fixed_frames(cfgs, factory) == base


def test_logs_carry_no_audio_fields(cfgs):
    _, _, summary = play_scripted(cfgs, True)
    assert not any(k in summary for k in ("sound", "audio", "volume", "mute", "music"))


# ---------------------------------------------------------------------------------------------- blind mode
def test_blind_mode_same_effects_for_every_opponent(cfgs):
    logs, bufs = {}, {}
    for enemy in ("dummy", "rusty"):
        sh, _ = start_match(cfgs, enemy=enemy, blind=True)
        sm = SoundManager(AUDIO, music=False, record=True)
        while sh.state == S.COUNTDOWN:
            sh.update(0.1, lambda: 0)
            sm.observe(sh)
        sh.fx = [("attack", 1), ("block", 0), ("dodge", 1), ("parry", 0), ("hit", 1), ("ko", 0)]
        sm.observe(sh)
        logs[enemy] = list(sm.log)
        bufs[enemy] = {k: v.get_raw() for k, v in sm.sounds.items()}
    assert logs["dummy"] == logs["rusty"] and bufs["dummy"] == bufs["rusty"]


def test_human_eval_music_off_by_default(monkeypatch):
    from scripts import human_eval as HE
    seen = []
    HE.main(["--participant", "P01", "--no-rating", "--out-dir", "tests/_tmp_audio_eval"],
            play_main=lambda argv: seen.append(argv) or None, input_fn=lambda *_: "")
    assert seen and all("--no-music" in a and "--music" not in a and "--blind" in a for a in seen)
    seen.clear()
    HE.main(["--participant", "P01", "--no-rating", "--music", "--out-dir", "tests/_tmp_audio_eval"],
            play_main=lambda argv: seen.append(argv) or None, input_fn=lambda *_: "")
    assert seen and all("--music" in a and "--no-music" not in a for a in seen)
    import shutil
    shutil.rmtree("tests/_tmp_audio_eval", ignore_errors=True)
