"""Game-free unit tests for the shared-memory client and the gym env."""

from __future__ import annotations

import logging
import struct
import threading
import time

import numpy as np
import pytest

from rainworld_rl import shared_memory as sm
from rainworld_rl.rainworld_env import RainWorldEnv
from rainworld_rl.shared_memory import ModState, SharedMemoryClient

from fake_mapping import FakeMapping


def make_client(mapping: FakeMapping, width: int = 8, height: int = 4) -> SharedMemoryClient:
    client = SharedMemoryClient(width, height, mapping_factory = lambda: mapping)
    client.slow_poll_interval = 0.001
    return client


# ---------------------------------------------------------------------------
# Header layout
# ---------------------------------------------------------------------------

def test_header_struct_matches_protocol_offsets():
    assert sm.HEADER_STRUCT.size == 64
    state = ModState(
        sync_flag = 2, ticks_per_step = 4, status = 0x3F,
        frame_width = 160, frame_height = 90, command = 1, command_result = 2, game_flags = 0x0B,
        heartbeat = 0xDEADBEEF, step_counter = 77, karma = 3, karma_cap = 5, food = 9, food_max = 11,
        player_x = 1.5, player_y = -2.5, room_index = -1, cycle_number = 12,
        action_bits = 0x1A5, cycle_progress = 0.5,
    )
    buf = state.pack()
    assert len(buf) == 64
    # Spot-check the documented offsets independently of the Struct.
    assert buf[0] == 2 and buf[1] == 0 and buf[2] == 4 and buf[3] == 0x3F   # offset 1 reserved
    assert struct.unpack_from("<I", buf, 4)[0] == 160
    assert struct.unpack_from("<I", buf, 8)[0] == 90
    assert buf[12] == 1 and buf[13] == 2 and buf[14] == 0x0B and buf[15] == 0
    assert struct.unpack_from("<I", buf, 16)[0] == 0xDEADBEEF
    assert struct.unpack_from("<I", buf, 20)[0] == 77
    assert buf[24:28] == bytes((3, 5, 9, 11))
    assert struct.unpack_from("<f", buf, 28)[0] == 1.5
    assert struct.unpack_from("<f", buf, 32)[0] == -2.5
    assert struct.unpack_from("<i", buf, 36)[0] == -1
    assert struct.unpack_from("<i", buf, 40)[0] == 12
    assert struct.unpack_from("<I", buf, 44)[0] == 0x1A5                   # action_bits
    assert struct.unpack_from("<f", buf, 48)[0] == 0.5
    assert buf[52:64] == bytes(12)


def test_header_pack_unpack_round_trip():
    state = ModState(
        sync_flag = 3, ticks_per_step = 255, status = 0x2A,
        frame_width = 1920, frame_height = 1080, command = 1, command_result = 1,
        heartbeat = 4_000_000_000, step_counter = 123456, karma = 9, karma_cap = 10, food = 7,
        player_x = 1024.25, player_y = 300.75, room_index = 2**31 - 1, cycle_number = -1,
        action_bits = 0x1FF,
    )
    again = ModState.unpack(state.pack())
    assert again == state
    assert again.player_dead is False
    assert again.connected is True
    assert again.ready is False
    assert again.human_override is True
    assert again.in_game is False
    assert again.mod_alive is True
    assert again.player_pos == (1024.25, 300.75)


def test_to_info_contains_expected_fields():
    state = ModState(status = sm.STATUS_PLAYER_DEAD | sm.STATUS_IN_GAME | sm.STATUS_READY,
                     karma = 2, karma_cap = 4, food = 1, player_x = 3.0, player_y = 4.0,
                     room_index = 5, cycle_number = 6, step_counter = 7,
                     food_max = 7, cycle_progress = 0.25, food_to_hibernate = 4, malnourished = 1,
                     game_flags = sm.GAME_FLAG_IN_SHELTER | sm.GAME_FLAG_CYCLE_SURVIVED)
    info = state.to_info()
    assert info == {
        "player_dead": True, "karma": 2, "karma_cap": 4, "food": 1, "food_max": 7,
        "player_pos": (3.0, 4.0), "room_index": 5, "cycle_number": 6,
        "step_counter": 7, "in_game": True, "ready": True, "human_override": False,
        "in_shelter": True, "cycle_survived": True, "rain": False, "dialog_open": False,
        "cycle_progress": 0.25, "food_to_hibernate": 4, "malnourished": True,
    }
    assert info["malnourished"] is True


def test_v3_game_state_fields_round_trip():
    state = ModState(food_max = 9, food_to_hibernate = 6, malnourished = 1, cycle_progress = 1.25,
                     game_flags = sm.GAME_FLAG_RAIN | sm.GAME_FLAG_DIALOG_OPEN)
    raw = state.pack()
    assert raw[sm.OFFSET_FOOD_MAX] == 9
    assert raw[sm.OFFSET_FOOD_TO_HIBERNATE] == 6
    assert raw[sm.OFFSET_MALNOURISHED] == 1
    assert raw[54:64] == bytes(10)                      # reserved tail untouched
    again = ModState.unpack(raw)
    assert again == state
    assert again.rain and again.dialog_open and not again.in_shelter and not again.cycle_survived
    assert again.cycle_progress == 1.25


def test_fake_mapping_writes_v3_fields():
    mapping = FakeMapping()
    mapping.mod_fields.update(food_max = 5, food_to_hibernate = 3, malnourished = 1, cycle_progress = 0.5,
                              game_flags = sm.GAME_FLAG_IN_SHELTER)
    mapping.next_step_game_flags = sm.GAME_FLAG_CYCLE_SURVIVED
    mapping.put(sm.OFFSET_SYNC_FLAG, bytes((sm.SYNC_ACTION_READY,)))
    mapping.service_step()
    info = mapping.header().to_info()
    assert (info["food_max"], info["food_to_hibernate"], info["malnourished"]) == (5, 3, True)
    assert info["cycle_progress"] == 0.5
    assert info["in_shelter"] and info["cycle_survived"] and not info["rain"]
    # the edge bit is one-shot
    mapping.put(sm.OFFSET_SYNC_FLAG, bytes((sm.SYNC_ACTION_READY,)))
    mapping.service_step()
    info = mapping.header().to_info()
    assert info["in_shelter"] and not info["cycle_survived"]


# ---------------------------------------------------------------------------
# Status bit ownership
# ---------------------------------------------------------------------------

def test_set_connected_preserves_mod_owned_bits():
    mapping = FakeMapping()
    mod_bits = sm.STATUS_PLAYER_DEAD | sm.STATUS_READY | sm.STATUS_HUMAN_OVERRIDE | sm.STATUS_IN_GAME | sm.STATUS_MOD_ALIVE
    mapping.set_status_byte(mod_bits)
    client = make_client(mapping)
    client._open()

    client.set_connected(True)
    assert mapping.header().status == mod_bits | sm.STATUS_CONNECTED

    # Mod flips some of its bits while we are connected...
    mapping.set_status_byte((mod_bits & ~sm.STATUS_PLAYER_DEAD) | sm.STATUS_CONNECTED)
    client.set_connected(False)
    assert mapping.header().status == (mod_bits & ~sm.STATUS_PLAYER_DEAD)
    assert sm.MOD_OWNED_STATUS_MASK & sm.PY_OWNED_STATUS_MASK == 0


def test_connect_sets_connected_and_dims_and_disconnect_clears():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    client = make_client(mapping, 32, 16)
    state = client.connect(liveness_timeout = 0.2)
    assert state.ready
    h = mapping.header()
    assert h.connected and h.mod_alive
    assert (h.frame_width, h.frame_height) == (32, 16)
    assert client.is_connected()

    client.disconnect()
    assert not mapping.header().connected
    assert mapping.header().mod_alive       # untouched
    assert mapping.closed
    assert not client.is_connected()


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

def test_connect_raises_when_heartbeat_static_and_mod_not_alive():
    mapping = FakeMapping(alive = False)
    client = make_client(mapping)
    with pytest.raises(sm.GameNotRunningError, match = "MOD_ALIVE clear"):
        client.connect(liveness_timeout = 0.1)
    assert mapping.closed
    assert not client.is_connected()
    assert not mapping.header().connected   # we never raised CONNECTED


def test_connect_raises_when_mod_alive_bit_set_but_heartbeat_static():
    mapping = FakeMapping(alive = False)
    mapping.set_mod_bit(sm.STATUS_MOD_ALIVE, True)
    client = make_client(mapping)
    with pytest.raises(sm.GameNotRunningError, match = "did not advance"):
        client.connect(liveness_timeout = 0.1)


def test_connect_ready_timeout():
    mapping = FakeMapping(auto_ready = False)  # alive, never READY
    client = make_client(mapping)
    with pytest.raises(sm.ReadyTimeoutError):
        client.connect(ready_timeout = 0.05, liveness_timeout = 0.1)
    # Still attached; caller may retry wait_for_ready.
    assert client.is_connected()


def test_connect_without_waiting_for_ready():
    mapping = FakeMapping()
    client = make_client(mapping)
    client.connect(wait_ready = False, liveness_timeout = 0.1)
    assert client.is_connected() and not mapping.header().ready


# ---------------------------------------------------------------------------
# Step handshake
# ---------------------------------------------------------------------------

def test_step_handshake_returns_frame_and_state():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    mapping.set_mod_bit(sm.STATUS_IN_GAME, True)
    mapping.fill_value = 200
    client = make_client(mapping, 8, 4)
    client.connect(liveness_timeout = 0.1)

    frame, state = client.step(action = sm.KEY_RIGHT | sm.KEY_JUMP, ticks_per_step = 4, timeout = 1.0)

    assert frame.shape == (4, 8, 3) and frame.dtype == np.uint8
    assert frame.flags.writeable
    assert np.all(frame == 200)
    assert mapping.steps_serviced == [(sm.KEY_RIGHT | sm.KEY_JUMP, 4)]
    assert mapping.header().sync_flag == sm.SYNC_IDLE       # python wrote IDLE after reading
    assert state.sync_flag == sm.SYNC_FRAME_READY           # snapshot taken at FRAME_READY
    assert state.step_counter == 1
    assert state.karma == 3 and state.karma_cap == 5 and state.food == 2
    assert state.player_pos == (123.5, -4.25)
    assert state.room_index == 42 and state.cycle_number == 1
    assert state.in_game and state.ready and not state.player_dead

    # Frame is a copy: the mod overwriting the mapping must not change it.
    mapping.fill_value = 1
    frame2, state2 = client.step(0, 1)
    assert np.all(frame == 200) and np.all(frame2 == 1)
    assert state2.step_counter == 2


def test_player_dead_edge_is_reported_from_status_bit():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    client = make_client(mapping)
    client.connect(liveness_timeout = 0.1)

    _, s1 = client.step(0)
    mapping.next_step_dead = True
    _, s2 = client.step(0)
    _, s3 = client.step(0)
    assert (s1.player_dead, s2.player_dead, s3.player_dead) == (False, True, False)


def test_step_timeout_when_mod_alive_but_not_servicing():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    mapping.service_steps = False
    client = make_client(mapping)
    client.connect(liveness_timeout = 0.1)
    with pytest.raises(sm.StepTimeoutError):
        client.step(0, timeout = 0.05)


def test_step_reports_game_gone_when_heartbeat_stops():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    client = make_client(mapping)
    client.connect(liveness_timeout = 0.1)
    mapping.alive = False  # game died
    with pytest.raises(sm.GameNotRunningError):
        client.step(0, timeout = 0.05)


def test_step_waits_through_human_override_without_timing_out(caplog):
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    mapping.set_mod_bit(sm.STATUS_HUMAN_OVERRIDE, True)
    client = make_client(mapping)
    client.slow_poll_interval = 0.005
    client.connect(liveness_timeout = 0.1)

    def release_later():
        time.sleep(0.3)
        mapping.set_mod_bit(sm.STATUS_HUMAN_OVERRIDE, False)

    t = threading.Thread(target = release_later)
    t.start()
    with caplog.at_level(logging.INFO, logger = "rainworld_rl.shared_memory"):
        started = time.monotonic()
        frame, state = client.step(sm.KEY_JUMP, timeout = 0.1)   # timeout << override duration
        elapsed = time.monotonic() - started
    t.join()

    assert elapsed >= 0.25
    assert frame.shape == (4, 8, 3)
    assert not state.human_override
    override_msgs = [r for r in caplog.records if "override active" in r.getMessage()]
    assert len(override_msgs) == 1 and override_msgs[0].levelno == logging.WARNING
    assert any("released" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def test_reset_command_is_acked():
    mapping = FakeMapping()
    client = make_client(mapping)
    client.connect(wait_ready = False, liveness_timeout = 0.1)
    # Stale result from a previous command must be cleared before issuing.
    mapping.put(sm.OFFSET_COMMAND_RESULT, bytes((sm.COMMAND_RESULT_OK,)))

    state = client.reset_game(timeout = 1.0)

    assert mapping.commands_received == [sm.COMMAND_RESET]
    assert state.command == sm.COMMAND_NONE
    assert state.command_result == sm.COMMAND_RESULT_OK
    assert state.ready


def test_reset_command_error_raises():
    mapping = FakeMapping()
    mapping.fail_commands = True
    client = make_client(mapping)
    client.connect(wait_ready = False, liveness_timeout = 0.1)
    with pytest.raises(sm.CommandError, match = "error"):
        client.reset_game(timeout = 1.0)


def test_reset_command_timeout_raises():
    mapping = FakeMapping()
    mapping.service_commands = False
    client = make_client(mapping)
    client.connect(wait_ready = False, liveness_timeout = 0.1)
    with pytest.raises(sm.CommandError, match = "not acknowledged"):
        client.reset_game(timeout = 0.05)


def test_reset_discards_stale_frame_ready():
    mapping = FakeMapping()
    client = make_client(mapping)
    client.connect(wait_ready = False, liveness_timeout = 0.1)
    mapping.put(sm.OFFSET_SYNC_FLAG, bytes((sm.SYNC_FRAME_READY,)))
    client.reset_game(timeout = 1.0)
    assert mapping.header().sync_flag == sm.SYNC_IDLE


def test_kill_player_command_is_acked_and_next_step_reports_death_edge():
    mapping = FakeMapping()
    client = make_client(mapping)
    client.connect(wait_ready = True, ready_timeout = 1.0, liveness_timeout = 0.1)
    mapping.put(sm.OFFSET_COMMAND_RESULT, bytes((sm.COMMAND_RESULT_OK,)))  # stale result is cleared first

    state = client.kill_player(timeout = 1.0)

    assert mapping.commands_received == [sm.COMMAND_KILL_PLAYER]
    assert state.command == sm.COMMAND_NONE
    assert state.command_result == sm.COMMAND_RESULT_OK
    assert mapping.steps_serviced == []            # the ack does not step the game

    _frame, s1 = client.step(0, 1, timeout = 1.0)
    _frame, s2 = client.step(0, 1, timeout = 1.0)
    assert s1.player_dead is True                  # edge on the step after the kill ...
    assert s2.player_dead is False                 # ... and only that step


def test_kill_player_error_raises():
    mapping = FakeMapping()
    mapping.fail_commands = True
    client = make_client(mapping)
    client.connect(wait_ready = True, ready_timeout = 1.0, liveness_timeout = 0.1)
    with pytest.raises(sm.CommandError, match = "error"):
        client.kill_player(timeout = 1.0)


def test_kill_player_rejected_when_not_ready():
    mapping = FakeMapping(auto_ready = False)      # mod never enters the game -> no player to kill
    client = make_client(mapping)
    client.connect(wait_ready = False, liveness_timeout = 0.1)
    with pytest.raises(sm.CommandError, match = "error"):
        client.kill_player(timeout = 1.0)
    assert mapping.commands_received == [sm.COMMAND_KILL_PLAYER]


def test_kill_player_timeout_raises():
    mapping = FakeMapping()
    mapping.service_commands = False
    client = make_client(mapping)
    client.connect(wait_ready = True, ready_timeout = 1.0, liveness_timeout = 0.1)
    with pytest.raises(sm.CommandError, match = "not acknowledged"):
        client.kill_player(timeout = 0.05)


def test_operations_require_connection():
    client = make_client(FakeMapping())
    with pytest.raises(sm.NotConnectedError):
        client.send_action(0)
    with pytest.raises(sm.NotConnectedError):
        client.wait_for_frame(0.01)
    with pytest.raises(sm.NotConnectedError):
        client.reset_game(0.01)
    with pytest.raises(sm.NotConnectedError):
        client.kill_player(0.01)


def test_frame_dimension_validation():
    with pytest.raises(ValueError):
        SharedMemoryClient(0, 10)
    with pytest.raises(ValueError):
        SharedMemoryClient(1921, 10)
    client = make_client(FakeMapping())
    with pytest.raises(ValueError):
        client.set_frame_dimensions(10, 1081)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def make_env(mapping: FakeMapping, **kwargs) -> RainWorldEnv:
    client = make_client(mapping, 8, 4)
    env = RainWorldEnv(8, 4, ticks_per_step = 2, frame_timeout = 0.5, reset_timeout = 1.0,
                       ready_timeout = 0.2, client = client, **kwargs)
    return env


def test_env_construction_is_cheap_and_never_touches_mapping():
    calls = []
    client = SharedMemoryClient(8, 4, mapping_factory = lambda: calls.append(1) or FakeMapping())
    env = RainWorldEnv(8, 4, client = client)
    assert calls == []
    assert not env.connected
    assert env.observation_space.shape == (4, 8, 3)
    import gymnasium.spaces as spaces
    assert isinstance(env.action_space, spaces.MultiBinary)
    assert env.action_space.shape == (sm.NUM_KEYS,) == (9,)


def test_env_reset_without_game_raises_clear_error():
    mapping = FakeMapping(alive = False)
    client = make_client(mapping)
    env = RainWorldEnv(8, 4, client = client)
    with pytest.raises(sm.GameNotRunningError) as excinfo:
        env.reset()
    msg = str(excinfo.value)
    assert "launch()" in msg
    assert "No running Rain World" in msg
    assert not env.connected


def test_env_reset_sends_reset_and_returns_frame_and_info():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_IN_GAME, True)
    env = make_env(mapping)

    obs, info = env.reset()

    assert mapping.commands_received == [sm.COMMAND_RESET]
    assert obs.shape == (4, 8, 3) and obs.dtype == np.uint8
    assert mapping.steps_serviced == [(0, 2)]         # one no-op step of ticks_per_step
    assert info["cycle_number"] == 0 and info["ready"] and info["in_game"]
    assert set(info) == {
        "player_dead", "karma", "karma_cap", "food", "player_pos", "room_index",
        "cycle_number", "step_counter", "in_game", "ready", "human_override",
        "food_max", "in_shelter", "cycle_survived", "rain", "dialog_open", "cycle_progress",
        "food_to_hibernate", "malnourished",
    }
    assert env.connected


def test_env_reset_wipe_false_skips_command_and_waits_ready():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    env = make_env(mapping)
    obs, info = env.reset(options = {"wipe": False})
    assert mapping.commands_received == []
    assert obs.shape == (4, 8, 3)


def test_env_reset_wipe_false_requires_ready():
    mapping = FakeMapping(auto_ready = False)  # never READY, and no RESET to make it so
    env = make_env(mapping)
    with pytest.raises(sm.ReadyTimeoutError):
        env.reset(options = {"wipe": False})


def test_env_step_never_terminates_and_reports_death_in_info():
    mapping = FakeMapping()
    env = make_env(mapping)
    env.reset()
    mapping.next_step_dead = True

    obs, reward, terminated, truncated, info = env.step(sm.KEY_JUMP)
    assert reward == 0.0 and terminated is False and truncated is False
    assert info["player_dead"] is True
    assert mapping.steps_serviced[-1] == (sm.KEY_JUMP, 2)

    _, _, _, _, info2 = env.step(0)
    assert info2["player_dead"] is False          # edge, not level
    assert info2["step_counter"] == 3
    assert env.last_state.step_counter == 3

    env.close()
    assert not env.connected
    assert not mapping.header().connected


def test_env_debug_kill_sends_command_and_death_edge_is_one_step():
    mapping = FakeMapping()
    env = make_env(mapping)
    env.reset()
    steps_before = len(mapping.steps_serviced)

    env.debug_kill()

    assert mapping.commands_received == [sm.COMMAND_RESET, sm.COMMAND_KILL_PLAYER]
    assert len(mapping.steps_serviced) == steps_before   # debug_kill() itself never steps

    _, _, terminated, truncated, info1 = env.step(0)
    _, _, _, _, info2 = env.step(0)
    assert info1["player_dead"] is True and terminated is False and truncated is False
    assert info2["player_dead"] is False                  # continual env: death is an edge, not an end


def test_env_debug_kill_requires_connection_and_propagates_command_error():
    env = make_env(FakeMapping())
    with pytest.raises(sm.GameNotRunningError):
        env.debug_kill()

    mapping = FakeMapping()
    env = make_env(mapping)
    env.reset()
    mapping.fail_commands = True
    with pytest.raises(sm.CommandError):
        env.debug_kill(timeout = 0.2)


def test_env_step_before_connect_raises():
    env = make_env(FakeMapping())
    with pytest.raises(sm.GameNotRunningError):
        env.step(0)


def test_env_set_frame_dimensions_updates_space_and_frames():
    mapping = FakeMapping()
    env = make_env(mapping)
    env.reset()
    env.set_frame_dimensions(6, 2)
    assert env.observation_space.shape == (2, 6, 3)
    obs, *_ = env.step(0)
    assert obs.shape == (2, 6, 3)
    assert (mapping.header().frame_width, mapping.header().frame_height) == (6, 2)


def test_gym_registration():
    import gymnasium as gym
    assert "RainWorld-v0" in gym.registry
    assert gym.registry["RainWorld-v0"].entry_point == "rainworld_rl.rainworld_env:RainWorldEnv"
