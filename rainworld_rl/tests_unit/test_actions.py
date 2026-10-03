"""Game-free tests for the raw-key action space and the optional Discrete(18) wrapper."""

from __future__ import annotations

import struct

import gymnasium.spaces as spaces
import numpy as np
import pytest

from rainworld_rl import shared_memory as sm
from rainworld_rl import wrappers
from rainworld_rl.rainworld_env import RainWorldEnv
from rainworld_rl.shared_memory import SharedMemoryClient
from rainworld_rl.wrappers import ACTION_NAMES, DISCRETE_ACTIONS, DiscreteActions

from fake_mapping import FakeMapping


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


def test_keys_vector_round_trip():
    for bits in (0, 1, sm.KEY_ALL_MASK, 0x155, 0x0AA, sm.KEY_DOWN | sm.KEY_RIGHT):
        vec = sm.bits_to_keys(bits)
        assert vec.shape == (sm.NUM_KEYS,) and vec.dtype == np.int8
        assert sm.keys_to_bits(vec) == bits
        assert sm.keys_to_bits(list(vec)) == bits
        assert sm.keys_to_bits(vec.astype(bool)) == bits
    with pytest.raises(ValueError):
        sm.keys_to_bits([1, 0, 1])
    with pytest.raises(ValueError):
        sm.keys_to_bits(np.zeros((2, sm.NUM_KEYS)))


def test_action_to_bits_accepts_ints_arrays_and_scalars():
    assert sm.action_to_bits(0) == 0
    assert sm.action_to_bits(sm.KEY_JUMP) == sm.KEY_JUMP
    assert sm.action_to_bits(np.int64(sm.KEY_MAP)) == sm.KEY_MAP
    assert sm.action_to_bits(np.array(sm.KEY_GRAB)) == sm.KEY_GRAB          # 0-d array
    assert sm.action_to_bits(True) == 1                                       # bool -> bit 0 (left)
    assert sm.action_to_bits(sm.bits_to_keys(0x1A5)) == 0x1A5
    assert sm.action_to_bits([0, 1, 0, 0, 1, 0, 0, 0, 0]) == sm.KEY_RIGHT | sm.KEY_JUMP
    assert sm.action_to_bits((1,) * sm.NUM_KEYS) == sm.KEY_ALL_MASK
    sample = spaces.MultiBinary(sm.NUM_KEYS).sample()
    assert sm.action_to_bits(sample) == sum(1 << i for i in range(sm.NUM_KEYS) if sample[i])
    with pytest.raises(ValueError):
        sm.action_to_bits(sm.KEY_ALL_MASK + 1)
    with pytest.raises(ValueError):
        sm.action_to_bits(-1)
    with pytest.raises(ValueError):
        sm.action_to_bits([1, 1])


def test_pressed_key_names():
    assert sm.pressed_key_names(0) == ()
    assert sm.pressed_key_names(sm.KEY_RIGHT | sm.KEY_JUMP | sm.KEY_SPECIAL) == ("right", "jump", "special")
    assert sm.pressed_key_names(sm.KEY_ALL_MASK) == sm.KEY_NAMES


# ---------------------------------------------------------------------------
# Client writes action_bits (offset 44), not the legacy byte
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
    assert mapping.raw(1, 1) == b"\x00"              # legacy byte untouched
    assert mapping.raw(sm.OFFSET_TICKS_PER_STEP, 1) == b"\x03"
    assert mapping.raw(sm.OFFSET_SYNC_FLAG, 1) == bytes((sm.SYNC_ACTION_READY,))


def test_client_send_action_accepts_vectors_and_key_names():
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    client = make_client(mapping)
    client.connect(liveness_timeout = 0.1)

    client.step(sm.bits_to_keys(sm.KEY_UP | sm.KEY_GRAB), ticks_per_step = 2)
    client.step([0, 0, 0, 0, 0, 0, 1, 0, 0])
    client.send_keys("right", "jump", ticks_per_step = 5)
    client.wait_for_frame(1.0)
    client.step()  # default = no keys

    assert mapping.steps_serviced == [
        (sm.KEY_UP | sm.KEY_GRAB, 2),
        (sm.KEY_THROW, 1),
        (sm.KEY_RIGHT | sm.KEY_JUMP, 5),
        (0, 1),
    ]


def test_fake_mapping_reads_action_bits_not_legacy_byte():
    """The fake mod (and the real one) must consume offset 44 only."""
    mapping = FakeMapping()
    mapping.set_mod_bit(sm.STATUS_READY, True)
    client = make_client(mapping)
    client.connect(liveness_timeout = 0.1)
    mapping.put(1, b"\x7f")  # garbage in the legacy slot must be ignored
    client.step(sm.KEY_MAP)
    assert mapping.steps_serviced == [(sm.KEY_MAP, 1)]


# ---------------------------------------------------------------------------
# Env: MultiBinary space, arrays and bitmasks
# ---------------------------------------------------------------------------

def make_env(mapping: FakeMapping, **kwargs) -> RainWorldEnv:
    client = make_client(mapping, 8, 4)
    return RainWorldEnv(8, 4, ticks_per_step = 2, frame_timeout = 0.5, reset_timeout = 1.0,
                        ready_timeout = 0.2, client = client, **kwargs)


def test_env_action_space_is_multibinary_over_keys():
    env = make_env(FakeMapping())
    assert isinstance(env.action_space, spaces.MultiBinary)
    assert env.action_space.shape == (sm.NUM_KEYS,)
    assert env.key_names == sm.KEY_NAMES
    sample = env.action_space.sample()
    assert env.action_space.contains(sample)
    assert env.action_space.contains(sm.bits_to_keys(sm.KEY_ALL_MASK))


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
    assert len(DISCRETE_ACTIONS) == len(ACTION_NAMES) == wrappers.NUM_DISCRETE_ACTIONS == 18
    assert ACTION_NAMES == (
        "No-op", "Left", "Right", "Up", "Down", "Jump", "Grab", "Throw",
        "Left+Jump", "Right+Jump", "Up+Jump", "Down+Jump",
        "Left+Grab", "Right+Grab", "Up+Grab", "Down+Grab",
        "Crawl Left", "Crawl Right",
    )
    for index, bits in EXPECTED_DISCRETE_BITS.items():
        assert wrappers.discrete_to_bits(index) == bits, ACTION_NAMES[index]
        assert sm.keys_to_bits(wrappers.discrete_to_keys(index)) == bits
    assert len(set(wrappers.DISCRETE_ACTION_BITS)) == 18
    # The discrete set never presses map or special.
    assert all(not (b & (sm.KEY_MAP | sm.KEY_SPECIAL)) for b in wrappers.DISCRETE_ACTION_BITS)
    with pytest.raises(ValueError):
        wrappers.discrete_to_bits(18)
    with pytest.raises(ValueError):
        wrappers.discrete_to_bits(-1)


def test_discrete_wrapper_exposes_discrete_18_and_forwards_key_vectors():
    mapping = FakeMapping()
    env = DiscreteActions(make_env(mapping))
    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == 18
    assert isinstance(env.unwrapped.action_space, spaces.MultiBinary)

    env.reset()
    for index in range(18):
        obs, reward, terminated, truncated, info = env.step(index)
        assert obs.shape == (4, 8, 3)
        assert reward == 0.0 and not terminated and not truncated
    assert [bits for bits, _t in mapping.steps_serviced[1:]] == [EXPECTED_DISCRETE_BITS[i] for i in range(18)]

    env.step(np.int64(9))  # numpy ints work too
    assert mapping.steps_serviced[-1][0] == sm.KEY_RIGHT | sm.KEY_JUMP

    for index in range(18):
        assert env.reverse_action(env.action(index)) == index
    with pytest.raises(ValueError):
        env.reverse_action(sm.bits_to_keys(sm.KEY_MAP))


def test_discrete_wrapper_rejects_non_key_envs():
    import gymnasium as gym

    class Dummy(gym.Env):
        action_space = spaces.Discrete(3)
        observation_space = spaces.Discrete(1)

    with pytest.raises(TypeError):
        DiscreteActions(Dummy())


def test_package_exports():
    import rainworld_rl
    for name in ("DiscreteActions", "ACTION_NAMES", "KEY_NAMES", "NUM_KEYS", "KEY_JUMP",
                 "encode_keys", "keys_to_bits", "bits_to_keys", "action_to_bits", "pressed_key_names"):
        assert name in rainworld_rl.__all__ and hasattr(rainworld_rl, name), name
