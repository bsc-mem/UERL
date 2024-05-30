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
import datetime as dt
import pickle
import os
import logging
import uuid
import yaml

from .agent import PER, duelingDQN, duelingDQNAgent


def read_config(file_path):
    """
    Read configuration file.

    Args:
        file_path (str): Path to configuration file.

    Returns:
        dict: Configuration.
    """
    _, file_extension = os.path.splitext(file_path)
    if not file_extension.lower() in ['.yaml', '.yml']:
        raise Exception('Configuration file must be a YAML file.')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    # Transform mitigation cost to timedelta
    config['mitigation_cost'] = dt.timedelta(minutes=config['mitigation_minutes'])
    return config


def check_directory(path):
    """
    Determine if path is a directory.

    Args:
        path (str): Path to directory.

    Returns:
        bool: True if path is a directory, False otherwise.
    """
    if not os.path.exists(path):
        raise Exception(f'Path {path} does not exist.')
    if os.path.isdir(path):
        return True
    raise Exception(f'Path {path} should be a directory.')


def check_pickle_file(path):
    """
    Determine if path is a pickle file.

    Args:
        path (str): Path to pickle file.

    Returns:
        bool: True if path is a pickle file, False otherwise.
    """
    if not os.path.exists(path):
        raise Exception(f'Path {path} does not exist.')
    _, file_extension = os.path.splitext(path)
    if file_extension.lower() in ['.pkl', '.pickle']:
        return True
    raise Exception(f'Path {path} should be a pickle file.')


def check_directory_or_pickle_file(path):
    """
    Determine if path is a directory or a pickle file.

    Args:
        path (str): Path to directory or pickle file.

    Returns:
        bool: True if path is a directory or a pickle file, False otherwise.
    """
    if not os.path.exists(path):
        raise Exception(f'Path {path} does not exist.')
    try:
        check_directory(path)
        return True
    except:
        pass
    try:
        check_pickle_file(path)
        return True
    except:
        raise Exception(f'Path {path} should be a directory or a pickle file.')
    

def setup_logging(verbose, debug):
    """
    Configure logging mode.

    Args:
        verbose (bool): Enable verbose mode.
        debug (bool): Enable debug mode.

    Returns:
        None
    """
    level = logging.WARNING
    if verbose and not debug:
        level = logging.INFO
    elif debug:
        level = logging.DEBUG
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format)


def load_data(fts_path, ues_path, jobs_path):
    """
    Load data from files: features, UEs, jobs.

    Args:
        fts_path (str): Path to features file.
        ues_path (str): Path to UEs file.
        jobs_path (str): Path to jobs file.

    Returns:
        tuple: (
            df (pd.DataFrame): Features dataframe.
            ues_df (pd.DataFrame): UEs dataframe.
            jobs_df (pd.DataFrame): Jobs dataframe.
        )
    """
    # Load features
    logging.info('Loading features data...')
    df = pd.read_csv(fts_path, parse_dates=['date_time'])
    df.set_index(['date_time', 'id_blade'], inplace=True)
    
    # Load UEs
    logging.info('Loading UEs data...')
    ues_df = pd.read_csv(ues_path)
    ues_df['date_time'] = pd.to_datetime(ues_df['date_time'])

    # Load jobs
    # Get extension of the file
    logging.info('Loading jobs data...')
    _, file_extension = os.path.splitext(jobs_path)
    if file_extension == '.csv':
        jobs_df = pd.read_csv(jobs_path)
        jobs_df['Elapsed'] = pd.to_timedelta(jobs_df['Elapsed'])
    elif file_extension == '.feather':
        jobs_df = pd.read_feather(jobs_path)
    else:
        raise Exception(f'Unkown {file_extension} extension for jobs DF file.')

    return df, ues_df, jobs_df


def datetime_splits(df, i_split, n_splits, train_prop):
    """
    Determine train/validation/test datetimes for this split.

    Args:
        df (pd.DataFrame): Features dataframe.
        i_split (int): Number specifying the ith split to be used for 
                       training/validation/testing.
        n_splits (int): Number of splits.
        train_prop (float): Proportion of data for training in this split.

    Returns:
        tuple: (
            train_dt (dt.datetime): Train end datetime.
            valid_dt (dt.datetime): Validation start datetime.
            test_dt (dt.datetime): Test end datetime.
        )
    """
    logging.info('Splitting train/validation/test...')

    # Decide datetime for train/validation/test for this split
    dt_min = df.index.get_level_values('date_time').min()
    dt_max = df.index.get_level_values('date_time').max()
    splits = get_splits(dt_min, dt_max , n_splits)
    if i_split == 0:
        # If first split use first 15 days for training and validation
        valid_dt = (dt_min + pd.Timedelta(days=15))
        train_dt = valid_dt
    else:
        valid_dt = splits[i_split]
        valid_part = ((valid_dt - splits[i_split - 1]) * train_prop)
        train_dt = splits[i_split - 1] + valid_part
    test_dt = splits[i_split + 1]

    logging.info(f'Train datetime: {train_dt}')
    logging.info(f'Validation datetime: {valid_dt}')
    logging.info(f'Test datetime: {test_dt}')
    return train_dt, valid_dt, test_dt


def df_splits(df, i_split, train_dt, valid_dt, test_dt):
    """
    Determine train/validation/test features dataframes for this split.

    Args:
        df (pd.DataFrame): Features dataframe.
        i_split (int): Number specifying the ith split to be used for 
                       training/validation/testing.
        train_dt (dt.datetime): Train end datetime.
        valid_dt (dt.datetime): Validation start datetime.
        test_dt (dt.datetime): Test end datetime.

    Returns:
        tuple: (
            df_train (pd.DataFrame): Train dataframe.
            df_valid (pd.DataFrame): Validation dataframe.
            df_test (pd.DataFrame): Test dataframe.
        )
    """
    # Compute DF train/validation/test sets for this split
    if 'date_time' in df.columns:
        dts_list = df['date_time']
    elif 'date_time' in df.index.names:
        dts_list = df.index.get_level_values('date_time')
    else:
        raise Exception('Could not find date_time column in dataframe.')
    df_train = df[dts_list < train_dt]
    df_test = df[(dts_list >= valid_dt) &
                 (dts_list < test_dt)]
    if i_split == 0:
        # If first split
        df_valid = df[(dts_list < valid_dt)]
    else:
        df_valid = df[(dts_list >= train_dt) &
                      (dts_list < valid_dt)]
    return df_train, df_valid, df_test


def ues_splits(ues_df, i_split, train_dt, valid_dt, test_dt):
    """
    Determine train/validation/test UEs dataframes for this split.

    Args:
        ues_df (pd.DataFrame): UEs dataframe.
        i_split (int): Number specifying the ith split to be used for 
                       training/validation/testing.
        train_dt (dt.datetime): Train end datetime.
        valid_dt (dt.datetime): Validation start datetime.
        test_dt (dt.datetime): Test end datetime.

    Returns:
        tuple: (
            ues_df_train (pd.DataFrame): Train dataframe.
            ues_df_valid (pd.DataFrame): Validation dataframe.
            ues_df_test (pd.DataFrame): Test dataframe.
        )
    """
    ues_df_train = ues_df[ues_df['date_time'] < train_dt]
    ues_df_test = ues_df[(ues_df['date_time'] >= valid_dt) &
                         (ues_df['date_time'] < test_dt)]
    if i_split == 0:
        ues_df_valid = ues_df[(ues_df['date_time'] < valid_dt)]
    else:
        ues_df_valid = ues_df[(ues_df['date_time'] >= train_dt) &
                              (ues_df['date_time'] < valid_dt)]
    return ues_df_train, ues_df_valid, ues_df_test


def get_train_freq(dt_min, dt_max, n_splits):
    """
    Get the train frequency.

    Args:
        dt_min (dt.datetime): Minimum datetime of the data.
        dt_max (dt.datetime): Maximum datetime of the data.
        n_splits (int): Number of splits.

    Returns:
        dt.timedelta: Train frequency.
    """
    secs_range = (dt_max - dt_min).total_seconds()
    return pd.Timedelta(days=(secs_range / (3600 * 24)) / n_splits)


def get_agent(train_env, hyparams, epsilon, epsilon_min, epsilon_decay):
    """
    Get RL agent.

    Args:
        train_env (gym.Env): Training environment.
        hyparams (dict): Hyperparameters.

    Returns:
        RL agent.
    """
    logging.info('Setting RL agent...')
    buffer = PER(hyparams['max_size_buffer'], burn_in=32,
                 PER_a=hyparams['per_a'], PER_b=hyparams['per_b'])
    dueling_dnnetwork = duelingDQN(train_env, device='cpu')
    agent = duelingDQNAgent(train_env, dueling_dnnetwork, buffer,
                            hyparams['learning_rate'], epsilon, epsilon_min, 
                            epsilon_decay, hyparams['batch_size'])
    # Set random ID for this agent's name
    agent_id = str(uuid.uuid1()).replace('-', '')[:10]
    logging.info(f'Agent ID: {agent_id}')
    agent.agent_id = agent_id
    return agent


def get_ue_cost(job_starts, job_n_nodes, job_hours,
                current_dt):
    """
    Get the UE cost in node-hours.

    Args:
        job_starts (list): Jobs start datetimes.
        job_n_nodes (list): Jobs number of nodes.
        job_hours (list): Jobs total node-hour costs.
        current_dt (dt.datetime): Current datetime.

    Returns:
        float: UE cost in node-hours.
    """
    # Compute node-hours since the last job start
    last_job_start_idx = np.where(pd.Series(job_starts) <= current_dt)[0][-1]

    if job_hours[last_job_start_idx] == 0:
        return 0

    last_job_start = job_starts[last_job_start_idx]
    last_job_n_nodes = job_n_nodes[last_job_start_idx]
    secs_job = (current_dt - last_job_start).total_seconds()
    ue_job_hours = (secs_job * last_job_n_nodes) / 3600
    return ue_job_hours


def get_hours_since_last_mitigation(job_starts, job_n_nodes,
                                    current_dt, miti_finni_times):
    """
    Get the hours since the last mitigation.

    Args:
        job_starts (list): Jobs start datetimes.
        job_n_nodes (list): Jobs number of nodes.
        current_dt (dt.datetime): Current datetime.
        miti_finni_times (list): Mitigations finnished datetimes.

    Returns:
        float: Hours since the last mitigation.
    """
    miti_finni_times = pd.to_datetime(miti_finni_times)
    hours_since_mitigation = np.inf
    is_prev_miti = current_dt >= miti_finni_times
    last_job_start_idx = np.where(pd.Series(job_starts) <= current_dt)[0][-1]
    last_job_n_nodes = job_n_nodes[last_job_start_idx]
    if any(is_prev_miti):
        # If there are mitigations before the UE
        finnished_dt = miti_finni_times[is_prev_miti][-1]
        secs_since_miti = (current_dt - finnished_dt).total_seconds()
        secs_cost = secs_since_miti * last_job_n_nodes
        hours_since_mitigation = secs_cost / 3600
    return hours_since_mitigation


def get_potential_ue_cost(job_starts, job_n_nodes, job_hours,
                          current_dt, miti_finni_times):
    """
    Get the potential UE cost in node-hours.

    Args:
        job_starts (list): Jobs start datetimes.
        job_n_nodes (list): Jobs number of nodes.
        job_hours (list): Jobs total node-hours cost.
        current_dt (dt.datetime): Current datetime.
        miti_finni_times (list): Mitigations finnished datetimes.

    Returns:
        float: Potential UE cost in node-hours.
    """
    # Potential ue cost in node-hours
    ue_job_hours = get_ue_cost(job_starts, job_n_nodes, job_hours,
                               current_dt)
    # Hours since last mitigation
    hours_since_mitigation = get_hours_since_last_mitigation(job_starts,
                                                             job_n_nodes,
                                                             current_dt,
                                                             miti_finni_times)
    return min(ue_job_hours, hours_since_mitigation)


def get_splits(dt_min, dt_max, n_splits):
    """
    Get the splits of the data.

    Args:
        dt_min (dt.datetime): Minimum datetime of the data.
        dt_max (dt.datetime): Maximum datetime of the data.
        n_splits (int): Number of splits.

    Returns:
        list: Data splits represented as datetimes.
    """
    # Add 2 days so the comparison of < with last split includes all data
    train_freq = get_train_freq(dt_min, dt_max, n_splits)
    curr_split = dt_min.floor('1d')
    splits = [curr_split.floor('1d')]

    while curr_split + train_freq < dt_max:
        curr_split += train_freq
        splits.append(curr_split.floor('1d'))
    # Replace last split because it's too short,
    # so we merge it with the previous one.
    # Add last split plus a day (cause we also floor with 1d)
    splits[-1] = dt_max.floor('1d') + pd.Timedelta('1d')
    return splits


def save_pickle(obj, fpath, fname):
    """
    Save a pickle object.

    Args:
        obj (object): Object to save.
        fpath (str): Path to save the pickle object.
        fname (str): Name of the pickle object.
    """
    # Create path if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    with open(os.path.join(fpath, fname), 'wb') as f:
        logging.info(f'Saving {fname}...')
        pickle.dump(obj, f)


def save_evaluation_results(results, output_path=None,
                            agent_id=None, eval_label=None):
    """
    Save evaluation results.

    Args:
        results (dict): Evaluation results.
        output_path (str, optional): Path to save the evaluation results.
                                     Defaults to None.
        agent_id (str, optional): Agent ID. Defaults to None. Ignored if None
                                  or file name is defined in output_path.
        eval_label (str, optional): Label to set by default in the file name.
                                    Only necessary when file name is not defined
                                    in output_path variable. Ignored otherwise.
    """
    no_file_name = output_path is None or output_path.isdir()
    no_necessary_vars = agent_id is None and eval_label is None
    if no_file_name and no_necessary_vars:
        raise Exception(('If a file name in output_path is not defined, '
                         'agent_id and eval_label must be defined.'))
    elif no_file_name:
        if agent_id is None:
            fname = f'{eval_label}.pkl'
        else:
            fname = f'{eval_label}_agent_{agent_id}.pkl'
    if output_path is not None:
        if output_path.isdir():
            results_path = os.path.join(output_path, fname)
        else:
            # Split output between directory and filename
            results_path = os.path.dirname(output_path)
            fname = os.path.basename(output_path)
    else:
        results_path = f'data/evaluations/{eval_label}/'
    save_pickle(results, results_path, fname)