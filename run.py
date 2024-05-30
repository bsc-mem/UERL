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


import numpy as np
import datetime as dt
import time
import argparse
import logging
from stable_baselines3.common.vec_env import DummyVecEnv

import src.common as comm
import src.evaluation as eval
from src.mn_environment import MNEnv


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Command line arguments and their values.
    """
    parser = argparse.ArgumentParser()
    helpers = {
        'i_split': 'Number specifying the ith split to be used for training.',
        'config': 'Path to configuration YAML file.',
        'output': ('Path tooutput. Can be either a directory or a pickle file. '
                   'If not specified, it will default to '
                   'evaluations/validation/validation_agent_{agent_id}.pkl.'),
        'verbose': 'Enable verbose mode.',
        'debug': 'Enable debug mode.',
    }

    parser.add_argument('-s', '--i_split', type=int,
                        help=helpers['i_split'], required=True)
    parser.add_argument('-c', '--config', type=str,
                        help=helpers['config'], required=True)
    parser.add_argument('-o', '--output', type=str, help=helpers['output'])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help=helpers['verbose'])
    parser.add_argument("-d", "--debug", action="store_true",
                        help=helpers['debug'])
    args = parser.parse_args()

    if args.output is not None:
        # Output should be a directory or pickle file
        comm.check_directory_or_pickle_file(args.output)

    # Configure logging
    comm.setup_logging(args.verbose, args.debug)

    logging.info('Arguments:')
    for k, v in vars(args).items():
        logging.info(f'\t{k}: {v}')

    return vars(args)


def check_hyperparams(hyperparams):
    """
    Check hyperparameters for RL agent.

    Args:
        hyperparams (dict): Hyperparameters.

    Returns:
        bool: True if hyperparameters are valid.
    """
    necessar_keys = [
        'learning_rate', 'batch_size', 'gamma', 'dnn_update_frequency',
        'dnn_sync_frequency', 'per_a', 'per_b', 'max_episodes_buffer',
        'max_size_buffer']
    for key in necessar_keys:
        if key not in hyperparams:
            raise Exception(f'Hyperparameter {key} not found in config file.')
    return True


def get_random_hyperparams():
    """
    Get random hyperparameters for RL agent.

    Returns:
        dict: Hyperparameters.
    """
    hyperparams = {
        'learning_rate': np.random.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        'batch_size': np.random.randint(1, 62),
        'gamma': round(np.random.uniform(0.10, 0.99), 2),
        'dnn_update_frequency': np.random.randint(1, 10),
        'dnn_sync_frequency': np.random.randint(2, 30),
        'per_a': round(np.random.uniform(0.5, 0.99), 2),
        'per_b': round(np.random.uniform(0.2, 0.6), 2),
        'max_episodes_buffer': np.random.randint(1000, 8000),
        'max_size_buffer': np.random.randint(10_000, 100_000),
    }
    logging.info(f'Hyperparameters: {hyperparams}')
    return hyperparams


def get_training_env(df_train, ues_df, jobs_df, mitigation_cost):
    """
    Get training gym environment.

    Args:
        df_train (pd.DataFrame): Train dataframe.
        ues_df (pd.DataFrame): UEs dataframe.
        jobs_df (pd.DataFrame): Jobs dataframe.
        mitigation_cost (dt.timedelta): Cost of a mitigation action.

    Returns:
        gym.Env: Training environment.
    """
    logging.info('Setting train environment...')
    train_env = MNEnv(df_train, ues_df, jobs_df, mitigation_cost,
                      is_training=True, jobs_max_iter=15_000)  # TODO do something with this?
    return DummyVecEnv([lambda: train_env])


def main():
    # Parse arguments
    args = parse_arguments()
    logging.info('Job starts running.')

    # Load configuration file
    config = comm.read_config(file_path=args['config'])
    logging.info(f'Config: {config}')

    # Load data
    df, ues_df, jobs_df = comm.load_data(config['fts_path'],
                                         config['ues_path'],
                                         config['jobs_path'])

    # Split data into train/validation/test
    logging.info(f'N splits: {config["n_splits"]}')
    train_dt, valid_dt, test_dt = comm.datetime_splits(df, args['i_split'],
                                                       config['n_splits'],
                                                       config['train_prop'])
    df_train, df_valid, _ = comm.df_splits(df, args['i_split'], train_dt,
                                           valid_dt, test_dt)
    _, ues_df_valid, _ = comm.df_splits(ues_df, args['i_split'], train_dt,
                                        valid_dt, test_dt)

    # Initialize hyperparameters
    if 'hyperparameters' in config:
        hyparams = config['hyperparameters']
        check_hyperparams(hyparams)
    else:
        # Random hyperparameters for hyperparam tuning
        hyparams = get_random_hyperparams()

    # Train environment
    train_env = get_training_env(df_train, ues_df, jobs_df,
                                 config['mitigation_cost'])
    # Initialize RL agent
    agent = comm.get_agent(train_env, hyparams, config['epsilon'],
                           config['epsilon_min'], config['epsilon_decay'])

    # Dict for storing training info
    info = {
        'n_episodes': config['n_episodes'],
        'i_split': args['i_split'],
        'epsilon': config['epsilon'],
        'epsilon_min': config['epsilon_min'],
        'epsilon_decay': config['epsilon_decay'],
    }
    # Add hyperparameteres to the info
    info.update(hyparams)

    # Train RL agent
    t0 = time.time()
    logging.info(f'Training...')
    agent.train(hyparams['gamma'],
                config['n_episodes'],
                hyparams['dnn_update_frequency'],
                hyparams['dnn_sync_frequency'])

    # Add training info
    info['train_time'] = time.time() - t0
    info['dueling_dnnetwork'] = agent.dueling_dnnetwork

    # Evaluate on validation data (the function stores the results)
    dt_min = df.index.get_level_values('date_time').min()
    logging.info(f'Running evaluation of validation set...')
    val_results = eval.run_evaluation(agent.dueling_dnnetwork,
                                      config['mitigation_cost'],
                                      args['i_split'], df_valid, ues_df_valid,
                                      dt_min, train_dt, valid_dt,
                                      config['job_samples_path'],
                                      is_validation=True)

    # Store validation results
    val_results.update(info)
    comm.save_evaluation_results(val_results, args['output'],
                                 agent.agent_id, 'validation')


if __name__ == '__main__':
    dt0 = dt.datetime.now()
    main()
    logging.info('Job ends running.')
    logging.info(f'Execution time: {dt.datetime.now() - dt0}')