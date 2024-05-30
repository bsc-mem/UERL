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
import os
import pickle


def jobs_distribution(jobs_df, dt_min, dt_max, max_sample=15_000):
    """
    Compute the distribution of jobs to be started in the given time frame.

    Args:
        jobs_df (pandas.DataFrame): Dataframe with jobs information.
        dt_min (datetime.datetime): Start time of the distribution.
        dt_max (datetime.datetime): End time of the distribution.
        max_sample (int, optional): Maximum number of jobs to be sampled.
                                    Defaults to 15,000.
    """
    nnodes = jobs_df['NNodes'].values
    # Elapsed times in seconds
    elapsed = jobs_df['Elapsed'].dt.total_seconds().values / 3600
    duration = elapsed * nnodes
    is_busy = (jobs_df['State'].values != 'IDLE')

    sample_idx = np.random.choice(np.arange(len(duration)), max_sample, 
                                  replace=True, p=nnodes/nnodes.sum())
    if isinstance(sample_idx, int) or isinstance(sample_idx, np.int64):
        # Cast to list in case a single index is returned
        sample_idx = [sample_idx]

    # TODO find more elegant way of doing this... At least adapt to total logs duration
    starts = pd.DataFrame({
        # Always start from the minimum datetime
        'start': jobs_df['Elapsed']\
                    .iloc[sample_idx]\
                    .cumsum()\
                    .shift(1, fill_value=pd.Timedelta('00:00:00')) + dt_min,
        'nnodes': nnodes[sample_idx],
        'hours_cost': elapsed[sample_idx] * nnodes[sample_idx] * is_busy[sample_idx]
    })
    # Filter out jobs that are not in the time frame
    starts = starts[starts['start'] < dt_max]
    return {
        'start': starts['start'].values,
        'nnodes': starts['nnodes'].values,
        'hours_cost': starts['hours_cost'].values}


def get_node_job_distributions(node_df, jobs_df, initial_dt, final_dt):
    """
    Get the job distribution for each node in the given time frame.

    Args:
        node_df (pandas.DataFrame): Dataframe with node information.
        jobs_df (pandas.DataFrame): Dataframe with jobs information.
        initial_dt (datetime.datetime): Start time of the distribution.
        final_dt (datetime.datetime): End time of the distribution.

    Returns:
        dict: Job distribution for each node, with job starts and number
              of nodes for each job.
    """
    # Consider node boots if they are less than 1 hour
    is_boot = node_df['hours_since_last_boot'] < 1
    node_boot_deltas = pd.to_timedelta(
        node_df[is_boot]['hours_since_last_boot'], 'h')
    # Node boots datetimes
    node_boots = (node_boot_deltas.index.get_level_values('date_time') - \
                  node_boot_deltas).dt.round('s')

    job_starts = []
    job_nnodes = []
    for i in range(len(node_boots)-1):
        if i == 0:
            # Force first datetime
            dt_min = initial_dt
        else:
            dt_min = node_boots.iloc[i]
        dt_max = node_boots.iloc[i+1]
        jobs = jobs_distribution(jobs_df, dt_min, dt_max)
        job_starts.extend(jobs['start'])
        job_nnodes.extend(jobs['nnodes'])

    # Last distribution (till the end)
    dt_min = node_boots.iloc[-1]
    if dt_min < final_dt:
        jobs = jobs_distribution(jobs_df, dt_min, final_dt)
        job_starts.extend(jobs['start'])
        job_nnodes.extend(jobs['nnodes'])

    # Dict for both starts and nnodes
    return {
        'start': np.array(job_starts),
        'nnodes': np.array(job_nnodes)}


def get_jobs_data(fpath=''):
    """
    Get the jobs data from the given file.

    Args:
        fpath (str, optional): File path of the jobs data. Defaults to ''.

    Returns:
        pandas.DataFrame: Jobs data.
    """
    # Get extension of the file
    ext = os.path.splitext(fpath)[-1]
    if ext == 'csv':
        jobs_df = pd.read_csv(fpath)
        jobs_df['Elapsed'] = pd.to_timedelta(jobs_df['Elapsed'])
    elif ext == 'feather':
        jobs_df = pd.read_feather(fpath)
    else:
        raise Exception('Unkown extension for jobs DF file.')
    return jobs_df


def get_job_distribution(jobs_file_path):
    """
    Get the job distribution for the given job index.

    Args:
        job_idx (int): Index of the job.
        jobs_path (str): Path to the job distributions.

    Returns:
        dict: Job distribution for the given job.
    """
    with open(jobs_file_path, 'rb') as f:
        return pickle.load(f)