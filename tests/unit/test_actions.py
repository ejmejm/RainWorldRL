"""Game-free tests for the raw-key action space and the optional Discrete(18) wrapper."""

from __future__ import annotations

import struct

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pytest

from rainworld_rl import shared_memory as sm
from rainworld_rl import wrappers
from rainworld_rl.rainworld_env import RainWorldEnv
from rainworld_rl.shared_memory import SharedMemoryClient
from rainworld_rl.wrappers import ACTION_NAMES, DiscreteActions

from tests.unit.fake_mapping import FakeMapping


# ---------------------------------------------------------------------------
# Key constants / encoding
# ---------------------------------------------------------------------------

def test_key_bit_layout_matches_protocol():
    """Bit i == KEY_NAMES[i]; values match PROTOCOL.md / SharedMemoryBridge.cs."""
    assert sm.KEY_NAMES == ("left", "right", "up", "down", "jump", "grab", "throw", "map", "special")
    assert sm.NUM_KEYS == 9
    assert sm.KEY_LEFT == 1 and sm.KEY_RIGHT == 2 and sm.KEY_UP == 4 and sm.KEY_DOWN == 8
    assert sm.KEY_JUMP == 16 and sm.KEY_GRAB == 32 and sm.KEY_THROW == 64
    assert sm.KEY_MAP == 128 and sm.KEY_SPECIAL == 256
    assert sm.KEY_ALL_MASK == 0x1FF
    for i, name in enumerate(sm.KEY_NAMES):
        assert sm.KEY_BITS[name] == 1 << i == getattr(sm, f"KEY_{name.upper()}")


def test_encode_keys_by_name_and_flag():
    assert sm.encode_keys() == 0
    assert sm.encode_keys("left", "jump") == sm.KEY_LEFT | sm.KEY_JUMP
    assert sm.encode_keys(right = True, grab = True, up = False) == sm.KEY_RIGHT | sm.KEY_GRAB
    assert sm.encode_keys("map", special = True) == sm.KEY_MAP | sm.KEY_SPECIAL
    with pytest.raises(KeyError):
        sm.encode_keys("pause")
    with pytest.raises(KeyError):
        sm.encode_keys(escape = True)
    assert sm.pressed_key_names(sm.KEY_RIGHT | sm.KEY_JUMP | sm.KEY_SPECIAL) == ("right", "jump", "special")


def test_keys_vector_and_action_to_bits_round_trip():
    for bits in (0, 1, sm.KEY_ALL_MASK, 0x155, 0x0AA, sm.KEY_DOWN | sm.KEY_RIGHT):
        vec = sm.bits_to_keys(bits)
        assert vec.shape == (sm.NUM_KEYS,) and vec.dtype == np.int8
        assert sm.keys_to_bits(vec) == sm.keys_to_bits(list(vec)) == sm.keys_to_bits(vec.astype(bool)) == bits
        assert sm.action_to_bits(vec) == sm.action_to_bits(list(vec)) == sm.action_to_bits(bits) == bits
    assert sm.action_to_bits(np.int64(sm.KEY_MAP)) == sm.KEY_MAP
    assert sm.action_to_bits(np.array(sm.KEY_GRAB)) == sm.KEY_GRAB          # 0-d array
    assert sm.action_to_bits(True) == 1                                       # bool -> bit 0 (left)
    for bad in ([1, 0, 1], np.zeros((2, sm.NUM_KEYS))):
        with pytest.raises(ValueError):
            sm.keys_to_bits(bad)
    for bad in (sm.KEY_ALL_MASK + 1, -1, [1, 1]):
        with pytest.raises(ValueError):
            sm.action_to_bits(bad)


# ---------------------------------------------------------------------------
# Client writes action_bits (offset 44)
# ---------------------------------------------------------------------------

def make_client(mapping: FakeMapping, width: int = 8, height: int = 4) -> SharedMemoryClient:
    client = SharedMemoryClient(width, height, mapping_factory = lambda: mapping)
    client.slow_poll_interval = 0.001
    return client


def test_client_writes_action_bits_at_offset_44():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    client = make_client(mapping)
    client.connect(liveness_timeout = 0.1)
    mapping.service_steps = False  # leave the header as written

    client.send_action(sm.KEY_LEFT | sm.KEY_DOWN | sm.KEY_SPECIAL, ticks_per_step = 3)

    assert struct.unpack_from("<I", mapping, sm.OFFSET_ACTION_BITS)[0] == sm.KEY_LEFT | sm.KEY_DOWN | sm.KEY_SPECIAL
    assert mapping.raw(1, 1) == b"\x00"              # reserved byte untouched
    assert mapping.raw(sm.OFFSET_TICKS_PER_STEP, 1) == b"\x03"
    assert mapping.raw(sm.OFFSET_SYNC_FLAG, 1) == bytes((sm.SYNC_ACTION_READY,))


# ---------------------------------------------------------------------------
# Env: MultiBinary space, arrays and bitmasks
# ---------------------------------------------------------------------------

def make_env(mapping: FakeMapping, **kwargs) -> RainWorldEnv:
    client = make_client(mapping, 8, 4)
    return RainWorldEnv(8, 4, ticks_per_step = 2, frame_timeout = 0.5, reset_timeout = 1.0,
                        ready_timeout = 0.2, client = client, **kwargs)


def test_env_step_accepts_vectors_bitmasks_and_samples():
    mapping = FakeMapping()
    env = make_env(mapping)
    env.reset()
    assert mapping.steps_serviced == [(0, 2)]  # reset's no-op step

    env.step(sm.bits_to_keys(sm.KEY_RIGHT))                    # ndarray vector
    env.step([1, 0, 0, 1, 0, 0, 0, 0, 0])                      # list vector (left + down)
    env.step(sm.KEY_JUMP | sm.KEY_THROW)                       # int bitmask
    env.step(np.int16(sm.KEY_SPECIAL))                         # numpy scalar bitmask
    sample = env.action_space.sample()
    env.step(sample)                                           # MultiBinary sample
    env.step(np.zeros(sm.NUM_KEYS, dtype = np.int8))           # explicit no-op vector

    assert [bits for bits, _ticks in mapping.steps_serviced[1:]] == [
        sm.KEY_RIGHT,
        sm.KEY_LEFT | sm.KEY_DOWN,
        sm.KEY_JUMP | sm.KEY_THROW,
        sm.KEY_SPECIAL,
        sm.keys_to_bits(sample),
        0,
    ]
    assert all(ticks == 2 for _bits, ticks in mapping.steps_serviced)


def test_env_step_rejects_bad_actions_before_touching_the_mod():
    mapping = FakeMapping()
    env = make_env(mapping)
    env.reset()
    n = len(mapping.steps_serviced)
    with pytest.raises(ValueError):
        env.step(sm.KEY_ALL_MASK + 1)
    with pytest.raises(ValueError):
        env.step([1, 0, 1])
    assert len(mapping.steps_serviced) == n


# ---------------------------------------------------------------------------
# Optional Discrete(18) wrapper
# ---------------------------------------------------------------------------

EXPECTED_DISCRETE_BITS = {
    0: 0,
    1: sm.KEY_LEFT,
    2: sm.KEY_RIGHT,
    3: sm.KEY_UP,
    4: sm.KEY_DOWN,
    5: sm.KEY_JUMP,
    6: sm.KEY_GRAB,
    7: sm.KEY_THROW,
    8: sm.KEY_LEFT | sm.KEY_JUMP,
    9: sm.KEY_RIGHT | sm.KEY_JUMP,
    10: sm.KEY_UP | sm.KEY_JUMP,
    11: sm.KEY_DOWN | sm.KEY_JUMP,
    12: sm.KEY_LEFT | sm.KEY_GRAB,
    13: sm.KEY_RIGHT | sm.KEY_GRAB,
    14: sm.KEY_UP | sm.KEY_GRAB,
    15: sm.KEY_DOWN | sm.KEY_GRAB,
    16: sm.KEY_LEFT | sm.KEY_DOWN,
    17: sm.KEY_RIGHT | sm.KEY_DOWN,
}


def test_discrete_table_maps_all_18_actions_to_the_right_bits():
    for index, bits in EXPECTED_DISCRETE_BITS.items():
        assert wrappers.discrete_to_bits(index) == bits, ACTION_NAMES[index]
        assert sm.keys_to_bits(wrappers.discrete_to_keys(index)) == bits
    with pytest.raises(ValueError):
        wrappers.discrete_to_bits(18)
    with pytest.raises(ValueError):
        wrappers.discrete_to_bits(-1)


def test_discrete_wrapper_forwards_key_vectors_and_rejects_non_key_envs():
    mapping = FakeMapping()
    env = DiscreteActions(make_env(mapping))
    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == 18

    env.reset()
    for index in range(18):
        env.step(index)
    assert [bits for bits, _t in mapping.steps_serviced[1:]] == [EXPECTED_DISCRETE_BITS[i] for i in range(18)]

    env.step(np.int64(9))  # numpy ints work too
    assert mapping.steps_serviced[-1][0] == sm.KEY_RIGHT | sm.KEY_JUMP

    for index in range(18):
        assert env.reverse_action(env.action(index)) == index
    with pytest.raises(ValueError):
        env.reverse_action(sm.bits_to_keys(sm.KEY_MAP))

    class Dummy(gym.Env):
        action_space = spaces.Discrete(3)
        observation_space = spaces.Discrete(1)

    with pytest.raises(TypeError):
        DiscreteActions(Dummy())
