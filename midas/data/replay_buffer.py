from typing import Union
from typing import Iterable, Optional
import jax
import gym
import gym.spaces
import h5py
import numpy as np
import pickle

import copy

from midas.data.dataset import Dataset, DatasetDict
from midas.utils.reproducibility import capture_numpy_rng, restore_numpy_rng
import collections
from flax.core import frozen_dict

def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int, ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity, ), dtype=np.float32)
        masks = np.empty((self.capacity, ), dtype=np.float32)
        discount = np.empty((self.capacity, ), dtype=np.float32)
        success_flag = np.zeros((self.capacity, ), dtype=np.float32)

        old_log_probs = np.zeros((self.capacity, ), dtype=np.float32)
        # Discounted Monte-Carlo return-to-go per query-level transition, used by
        # offline analysis. Defaults to zero for algorithms that never set it.
        mc_returns = np.zeros((self.capacity, ), dtype=np.float32)

        self.data = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'next_actions': next_actions,
            'rewards': rewards,
            'masks': masks,
            'discount': discount,
            'success_flag': success_flag,
            'old_log_probs': old_log_probs,
            'mc_returns': mc_returns,
        }

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = dict()
        self.streaming_buffer_size = None # this is for streaming the online data
        self._last_traj_indices = None  # indices of the most recently inserted trajectory

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        self.traj_bounds[self._traj_counter] = (self._start, self.size) # [start, end)
        self._last_traj_indices = np.arange(self._start, self.size)
        self._start = self.size
        self._traj_counter += 1

    def reset(self):
        """Drop all stored transitions without reallocating ``self.data``.

        Used by training to turn the main buffer into a pure
        online buffer after demos have been copied into the demo/offline
        buffer: the backing arrays are reused and simply overwritten as new
        ``insert`` calls advance ``self.size`` from zero.
        """
        self.size = 0
        self._start = 0
        self._traj_counter = 0
        self.traj_bounds = dict()
        self._last_traj_indices = None

    def _recompute_mc_returns(self):
        """Fill ``self.data['mc_returns']`` with discounted return-to-go.

        Backfill path for buffers restored from snapshots that predate the
        ``mc_returns`` field. Returns are accumulated backward within each
        trajectory using the per-transition ``discount`` already stored
        (``discount`` is ``gamma ** query_freq`` at the query-transition level).
        """
        rewards = self.data['rewards']
        discount = self.data['discount']
        mc = self.data['mc_returns']
        for start, end in self.traj_bounds.values():
            running = 0.0
            for t in range(end - 1, start - 1, -1):
                running = rewards[t] + discount[t] * running
                mc[t] = running

    def get_random_trajs(self, num_trajs: int):
        if hasattr(self.np_random, "integers"):
            self.which_trajs = self.np_random.integers(0, self._traj_counter, num_trajs)
        else:
            self.which_trajs = self.np_random.randint(0, self._traj_counter, num_trajs)
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []
        discount_list = []

        for i in self.which_trajs:
            start, end = self.traj_bounds[i]
            
            # handle this as a dictionary
            obs_dict_curr_traj = dict()
            for k in self.data['observations']:
                obs_dict_curr_traj[k] = self.data['observations'][k][start:end]
            observations_list.append(obs_dict_curr_traj)
            
            next_obs_dict_curr_traj = dict()
            for k in self.data['next_observations']:
                next_obs_dict_curr_traj[k] = self.data['next_observations'][k][start:end]    
            next_observations_list.append(next_obs_dict_curr_traj)
            
            actions_list.append(self.data['actions'][start:end])
            rewards_list.append(self.data['rewards'][start:end])
            terminals_list.append(1-self.data['masks'][start:end])
            masks_list.append(self.data['masks'][start:end])


        
        batch = {
            'observations': observations_list,
            'next_observations': next_observations_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'terminals': terminals_list,
            'masks': masks_list,
            
            
        }
        return batch
        
    def _grow_capacity(self, min_extra: int):
        """Ensure ``self.capacity - self.size >= min_extra`` by doubling as needed.

        ``insert`` historically doubled exactly once when full; ``append_delta``
        can add a chunk longer than ``self.capacity``, so the loop is required
        to avoid silent overflow.
        """
        while self.size + min_extra > self.capacity:
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            rewards = np.empty((self.capacity, ), dtype=np.float32)
            masks = np.empty((self.capacity, ), dtype=np.float32)
            discount = np.empty((self.capacity, ), dtype=np.float32)
            success_flag = np.zeros((self.capacity, ), dtype=np.float32)
            old_log_probs = np.zeros((self.capacity, ), dtype=np.float32)
            mc_returns = np.zeros((self.capacity, ), dtype=np.float32)

            data_new = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'next_actions': next_actions,
                'rewards': rewards,
                'masks': masks,
                'discount': discount,
                'success_flag': success_flag,
                'old_log_probs': old_log_probs,
                'mc_returns': mc_returns,
            }

            for x in data_new:
                if isinstance(self.data[x], np.ndarray):
                    self.data[x] = np.concatenate((self.data[x], data_new[x]), axis=0)
                elif isinstance(self.data[x], dict):
                    for y in self.data[x]:
                        self.data[x][y] = np.concatenate((self.data[x][y], data_new[x][y]), axis=0)
                else:
                    raise TypeError()
            self.capacity *= 2

    def insert(self, data_dict: DatasetDict):
        self._grow_capacity(1)

        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][self.size] = data_dict[x][y]
                else:                        
                    self.data[x][self.size] = data_dict[x]
        self.size += 1
    
    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']
        self.data['next_actions'] = (self.data['next_actions'] - action_stats['mean']) / action_stats['std']

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if indx is not None:
            indices = indx
        elif self.streaming_buffer_size:
            if hasattr(self.np_random, "integers"):
                indices = self.np_random.integers(0, self.streaming_buffer_size, batch_size)
            else:
                indices = self.np_random.randint(0, self.streaming_buffer_size, batch_size)
        else:
            if hasattr(self.np_random, "integers"):
                indices = self.np_random.integers(0, self.size, batch_size)
            else:
                indices = self.np_random.randint(0, self.size, batch_size)
        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
            else:
                raise TypeError()
        
        return frozen_dict.freeze(data_dict)

    def sample_from_last_traj(self, batch_size: int) -> frozen_dict.FrozenDict:
        """Sample a batch from the most recently inserted trajectory.
        
        Samples with replacement from the last trajectory's indices.
        If the trajectory has fewer transitions than batch_size, transitions
        will be repeated.
        
        Returns:
            FrozenDict batch sampled from the last trajectory.
        """
        assert self._last_traj_indices is not None, "No trajectory has been inserted yet"
        indices = self.np_random.choice(self._last_traj_indices, size=batch_size, replace=True)
        return self.sample(batch_size, indx=indices)

    def get_rng_state(self):
        """Return a portable snapshot of this buffer's private sampler RNG."""

        return capture_numpy_rng(self.np_random)

    def set_rng_state(self, state) -> None:
        """Restore a state returned by :meth:`get_rng_state`."""

        self._np_random = restore_numpy_rng(getattr(self, '_np_random', None), state)

    def get_last_traj_indices(self) -> Optional[np.ndarray]:
        """Return the indices of the most recently inserted trajectory."""
        return self._last_traj_indices

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


    def save(self, filename):
        save_dict = dict(
            data=self.data,
            size=self.size,
            _traj_counter=self._traj_counter,
            _start=self._start,
            traj_bounds=self.traj_bounds,
            _last_traj_indices=self._last_traj_indices,
            rng_state=self.get_rng_state(),
        )
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)


    def restore(self, filename):
        with open(filename, 'rb') as f:
            save_dict = pickle.load(f)
        self.data = save_dict['data']
        self.size = save_dict['size']
        self._traj_counter = save_dict['_traj_counter']
        self._start = save_dict['_start']
        self.traj_bounds = save_dict['traj_bounds']
        self._last_traj_indices = save_dict.get('_last_traj_indices', None)
        if 'rng_state' in save_dict:
            self.set_rng_state(save_dict['rng_state'])
        # Backfill mc_returns for snapshots written before the field existed so
        # downstream samplers always find the key.
        if 'mc_returns' not in self.data:
            self.data['mc_returns'] = np.zeros_like(self.data['rewards'])
            self._recompute_mc_returns()

    def _slice_data(self, start: int, end: int):
        """Slice every key in ``self.data`` over ``[start, end)``.

        Recurses into nested dicts (e.g. ``observations``) so adding a new field
        to ``self.data`` does not silently drop on save.
        """
        sliced = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                sliced[k] = v[start:end].copy()
            elif isinstance(v, dict):
                sliced[k] = {sk: sv[start:end].copy() for sk, sv in v.items()}
            else:
                raise TypeError(f"unexpected type for self.data[{k!r}]: {type(v)}")
        return sliced

    def _assign_chunk(self, chunk: dict, offset: int, length: int):
        """Write ``chunk`` (per-key arrays of equal leading dim ``length``) into
        ``self.data[offset:offset+length]``.

        Only assigns keys present in both ``chunk`` and ``self.data`` so a delta
        produced by an older code version with fewer keys still loads cleanly.
        """
        for k, v in self.data.items():
            if k not in chunk:
                continue
            chunk_v = chunk[k]
            if isinstance(v, np.ndarray):
                v[offset:offset + length] = chunk_v
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if sk in chunk_v:
                        sv[offset:offset + length] = chunk_v[sk]

    def save_delta(self, filename: str, since_traj_id: int) -> int:
        """Serialize trajectories ``[since_traj_id, self._traj_counter)`` to HDF5.

        Trajectories are stored contiguously between ``_start`` and ``size``,
        so the on-disk layout is one HDF5 dataset per buffer key carrying a
        single contiguous slice (nested ``observations`` / ``next_observations``
        live in HDF5 groups). Per-trajectory boundaries are reconstructed on
        load from the ``traj_lengths`` dataset. HDF5 sidesteps Python pickle's
        per-array protocol overhead and lets us write/read raw numpy buffers
        directly — measured ~1.3x faster write and ~1.8x faster read than
        pickle on a 62 MB delta.

        Returns the new high-water mark (``self._traj_counter``). Raises if
        ``since_traj_id`` is missing from ``traj_bounds`` (would indicate a
        save triggered mid-trajectory, which the training loop disallows).
        """
        end_traj_id = self._traj_counter
        if since_traj_id == end_traj_id:
            since_offset = self.size
        else:
            if since_traj_id not in self.traj_bounds:
                raise RuntimeError(
                    f"save_delta: traj_bounds missing id={since_traj_id} (have "
                    f"{sorted(self.traj_bounds.keys())[:5]}..); buffer is not at "
                    f"a trajectory boundary."
                )
            since_offset = self.traj_bounds[since_traj_id][0]
        end_offset = self.size
        traj_lengths = np.array(
            [self.traj_bounds[tid][1] - self.traj_bounds[tid][0]
             for tid in range(since_traj_id, end_traj_id)],
            dtype=np.int64,
        )
        with h5py.File(filename, 'w') as f:
            f.attrs['format_version'] = 2
            f.attrs['schema'] = 'keywise'
            f.attrs['since_traj_id'] = since_traj_id
            f.attrs['end_traj_id'] = end_traj_id
            f.create_dataset('traj_lengths', data=traj_lengths)
            data_group = f.create_group('data')
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    data_group.create_dataset(key, data=value[since_offset:end_offset])
                elif isinstance(value, dict):
                    sub = data_group.create_group(key)
                    for sk, sv in value.items():
                        sub.create_dataset(sk, data=sv[since_offset:end_offset])
                else:
                    raise TypeError(
                        f"save_delta: unsupported type for self.data[{key!r}]: {type(value)}"
                    )
        return end_traj_id

    def append_delta(self, filename: str) -> None:
        """Load a delta produced by ``save_delta`` and append it.

        Each on-disk dataset is read with a direct numpy slice assignment
        into ``self.data`` starting at the current ``self.size`` (one bulk
        copy per key — no per-trajectory Python loop), then ``traj_bounds``
        is rebuilt from the per-trajectory lengths. New ``traj_bounds`` keys
        are the live ``self._traj_counter`` (not the on-disk traj id), so
        chains composed from independent runs never collide.
        """
        with h5py.File(filename, 'r') as f:
            traj_lengths = f['traj_lengths'][...]
            total_len = int(traj_lengths.sum()) if traj_lengths.size else 0
            if total_len == 0:
                return
            self._grow_capacity(total_len)
            offset = self.size
            data_group = f['data']
            for key, value in self.data.items():
                if key not in data_group:
                    continue
                obj = data_group[key]
                if isinstance(value, np.ndarray) and isinstance(obj, h5py.Dataset):
                    obj.read_direct(value, np.s_[:total_len], np.s_[offset:offset + total_len])
                elif isinstance(value, dict) and isinstance(obj, h5py.Group):
                    for sk, sv in value.items():
                        if sk in obj:
                            obj[sk].read_direct(sv, np.s_[:total_len], np.s_[offset:offset + total_len])
        self.size = offset + total_len
        cursor = offset
        last_chunk_offset = None
        for chunk_len in traj_lengths.tolist():
            self.traj_bounds[self._traj_counter] = (cursor, cursor + chunk_len)
            self._traj_counter += 1
            last_chunk_offset = cursor
            cursor += chunk_len
        self._start = self.size
        if last_chunk_offset is not None:
            self._last_traj_indices = np.arange(last_chunk_offset, self.size)
