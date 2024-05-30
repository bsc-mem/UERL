# SPDX-License-Identifier: BSD-3-Clause
# 
# This file is part of UERL project (https://github.com/bsc-mem/UERL).
# 
# Copyright (c) 2024 Isaac Boixaderas Coderch
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the conditions stated in the BSD-3-Clause License are met.
# 
# For more information, see the LICENSE file at the root of this project or visit
# https://opensource.org/licenses/BSD-3-Clause.


import pandas as pd
import numpy as np
import logging
import gym
from gym import spaces

from . import job_funs
from . import common as comm


class MNEnv(gym.Env):
    """Custom Environment that follows the gym interface."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, ues_df, jobs_df, mitigation_cost,
                 is_training=True, jobs_max_iter=None):
        """
        Initialize the environment.

        Args:
            df (pd.DataFrame): Dataframe with all the features data.
            ues_df (pd.DataFrame): Dataframe with UEs data.
            jobs_df (pd.DataFrame): Dataframe with jobs data.
            mitigation_cost (dt.timedelta): Cost of a mitigation action.
            is_training (bool, optional): If the environment is in training mode.
                                          Defaults to True.
            jobs_max_iter (int, optional): Maximum number of jobs to be 
                                           started in the distribution.
                                           Defaults to None.
        """
        super(MNEnv, self).__init__()
        # Copy data: full_df is the original dataframe, whereas df is 
        # for a specific node, set when resetting the environment
        self.full_df = df.copy()
        self.df = None
        self.all_dtimes = self.full_df.index.get_level_values('date_time')
        self.all_id_blades = self.full_df.index.get_level_values('id_blade')
        self.id_blades = self.all_id_blades.unique()
        # Cost of performing mitigation
        self.mitigation_cost = mitigation_cost
        # Current step
        self.current_step = 0
        # Date times when mitigations are requested
        self.miti_req_times = np.array([])
        # Date times when mitigations are finnished
        self.miti_finni_times = np.array([])
        # Current node jobs during the whole production period
        self.current_job_hours = np.array([])
        # Actions: do nothing (0) or mitigation (1)
        self.action_space = spaces.Discrete(2)
        # Observation space
        # Add 1 extra feature automatically calculated: cost in case of UE
        shape = (1, 1, len(self.full_df.columns) + 1)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=shape)
        self.jobs_df = jobs_df
        # When training, the reward is adapted for learning.
        # Otherwise, the reward is the real cost
        self.is_training = is_training
        # TODO: decide what to do with this
        self.jobs_max_iter = jobs_max_iter
        # Load UEs data only for the time of the logs data
        ues_df = ues_df[(ues_df['date_time'] >= self.all_dtimes.min()) &
                        (ues_df['date_time'] <= self.all_dtimes.max())]
        # Optimization: Precalculate when UEs are going to take place
        self.ues_df = pd.DataFrame()
        self.ue_idxs = []
        for _, ue in ues_df.iterrows():
            filtered = self.full_df[
                (self.all_dtimes <= ue['date_time'].ceil('1min')) &
                (self.all_id_blades == ue['id_blade'])]
            if not filtered.empty:
                self.ue_idxs.append(filtered.iloc[-1].name)
                # Convert to DF and transpose `ue` for proper concatenation
                self.ues_df = pd.concat(
                    [self.ues_df, ue.to_frame().T], axis=0, ignore_index=True)

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple: The observation, reward, done, and info.
        """
        curr_time = self._current_time()
        logging.debug((
            f'ID blade: {self._current_blade()}; '
            f'Current time: {curr_time}; '
            f'Action: {action}; '
            f'Current step: {self.current_step}; '
            f'Current job hours: {self.current_job_hours}; '
        ))

        reward = 0
        if action == 1:
            if curr_time not in self.miti_req_times:
                self.miti_req_times = np.append(self.miti_req_times, curr_time)
                delayed = curr_time + self.mitigation_cost
                self.miti_finni_times = np.append(self.miti_finni_times, delayed)
            reward -= self.mitigation_cost.total_seconds() / 3600
        
        # Update reward if there is an UE during this step
        curr_idx = self.df.iloc[self.current_step].name
        if curr_idx in self.ue_idxs:
            current_ue = self.ues_df.iloc[self.ue_idxs.index(curr_idx)]
            potential_ue_cost = comm.get_potential_ue_cost(
                                        self.job_starts['start'],
                                        self.job_starts['nnodes'],
                                        self.job_starts['hours_cost'],
                                        current_ue['date_time'],
                                        self.miti_finni_times)
            reward -= potential_ue_cost
        
        self.current_step += 1
        done = self.current_step == len(self.df)
        
        if not done:
            # Update next job hours in (maybe new or previous) blade
            self.current_job_hours = self._current_job_size()
            curr_state = self._current_state()
            logging.debug(f'DONE - Current state: {curr_state}')
        else:
            curr_state = pd.Series(np.zeros(len(self.df.columns)))
        
        info = {'miti_finni_times': self.miti_finni_times}
        return curr_state, reward, done, info
    
    def reset(self):
        """
        Reset the state of the environment to an initial state.

        Returns:
            np.ndarray: The initial observation of the environment.
        """
        # Select only one of the nodes randomly
        self.id_blade = np.random.choice(self.id_blades)
        self.df = self.full_df[self.all_id_blades == self.id_blade]
        # Current step
        self.current_step = 0
        self.current_dt = self._current_time()
        self.last_dt = self.current_dt
        # Date times when mitigations were requested
        self.miti_req_times = np.array([])
        # Date times when mitigations were finnished
        self.miti_finni_times = np.array([])
        # Current job sizes
        self.job_starts = self._get_job_distributions()
        self.current_job_hours = self._current_job_size()
        return self._current_state()

    def _current_state(self):
        """
        Decide job size randomly sampled from a real job distribution.

        Returns:
            np.ndarray: The current state of the environment.
        """
        # Time difference between last mitigation and current time
        state = self.df.iloc[self.current_step].copy()
        curr_time = self._current_time()

        # Calculate the hours since the last mitigation request
        hours_since_miti_req = np.inf
        if len(self.miti_req_times) > 0:
            dt_req = self.miti_req_times[-1]
            hours_since_miti_req = (curr_time - dt_req).total_seconds() / 3600
        state['hours_since_mitigation_req'] = min(self.current_job_hours,
                                                  hours_since_miti_req)
        return state

    def _current_blade(self):
        """
        Get the id of the current blade.

        Returns:
            int: The id of the current blade.
        """
        return self.id_blade

    def _current_time(self):
        """
        Get the current time.

        Returns:
            dt.datetime: The current time in the environment.
        """
        return self.all_dtimes[self.current_step]

    def _current_job_size(self):
        """
        Get the current job size calculated as the number of seconds 
        elapsed since the job started times the number of nodes times 
        the cost multiplier.

        Returns:
            int: The current job size.
        """
        curr_time = self._current_time()
        prev_jobs = pd.Series(self.job_starts['start']) <= curr_time
        last_job_starts = np.where(prev_jobs)[0]
        if len(last_job_starts) == 0:
            err = 'IndexError: index -1 is out of bounds for axis 0 with size 0.'
            raise Exception(err)
        last_job_start_idx = last_job_starts[-1]
        last_job_start_dt = self.job_starts['start'][last_job_start_idx]
        last_job_n_nodes = self.job_starts['nnodes'][last_job_start_idx]
        # Seconds of the current running job
        job_secs = ((curr_time - last_job_start_dt).total_seconds() / 3600)
        return job_secs * last_job_n_nodes

    def _get_job_distributions(self):
        """
        Get the job distribution for the current blade.

        Returns:
            pd.DataFrame: The job distribution for the current blade.
        """
        # List of the jobs' start datetimes and number of nodes used
        initial_dt = self.all_dtimes.min()
        final_dt = self.all_dtimes.max()
        return job_funs.jobs_distribution(self.jobs_df, initial_dt,
                                          final_dt, self.jobs_max_iter)

    def render(self, mode='human', close=False):
        """
        Display the environment.
        """
        print(self._current_time())
        print('State:', np.array(self._current_state()))
