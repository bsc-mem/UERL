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
import os
import pickle
import argparse
import logging
from stable_baselines3.common.vec_env import DummyVecEnv

import src.common as comm
import src.evaluation as eval


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Command line arguments and their values.
    """
    parser = argparse.ArgumentParser()
    helpers = {
        'config': 'Path to configuration YAML file.',
        'input': 'Path to input directory with the validation files.',
        'output': ('Path tooutput. Can be either a directory or a pickle file. '
                   'If not specified, it will default to '
                   'evaluations/test/test_agent_{agent.agent_id}.pkl.'),
        'verbose': 'Enable verbose mode.',
        'debug': 'Enable debug mode.',
    }

    parser.add_argument('-c', '--config', type=str,
                        help=helpers['config'], required=True)
    parser.add_argument('-i', '--input', type=str,
                        help=helpers['output'], required=True)
    parser.add_argument('-o', '--output', type=str, help=helpers['output'])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help=helpers['verbose'])
    parser.add_argument("-d", "--debug", action="store_true",
                        help=helpers['debug'])
    args = parser.parse_args()

    # Input must be a directory
    comm.check_directory(args.input)
    if args.output is not None:
        # Output must be a directory or a pickle file
        comm.check_directory_or_pickle_file(args.output)

    # Configure logging
    comm.setup_logging(args.verbose, args.debug)

    logging.info('Arguments:')
    for k, v in vars(args).items():
        logging.info(f'\t{k}: {v}')

    return vars(args)


def main():
    # Parse arguments
    args = parse_arguments()
    logging.info('Job starts running.')

    # Load configuration file
    config = comm.read_config(file_path=args['config'])

    # Load data
    df, ues_df, _ = comm.load_data(config['fts_path'],
                                   config['ues_path'],
                                   config['jobs_path'])

    # Get best validation results for each split
    best_validations = {}
    for val_file in os.listdir(args['input']):
        logging.info(f'Processing validation file: {val_file}')
        val_path = os.path.join(args['input'], val_file)
        comm.check_pickle_file(val_path)
        with open(val_path, 'rb') as f:
            val_results = pickle.load(f)

        # Update best validation if needed
        avg_cost = val_results['avg_ue_cost'] + val_results['avg_mitigation_cost']
        if val_results['i_split'] not in best_validations:
            best_validations[val_results['i_split']] = {
                'avg_cost': avg_cost,
                'val_results': val_results}
        else:
            if avg_cost < best_validations[val_results['i_split']]['avg_cost']:
                best_validations[val_results['i_split']] = {
                    'avg_cost': avg_cost,
                    'val_results': val_results}

    # Run test on the best validation agent for each split
    test_results = []
    dt_min = df.index.get_level_values('date_time').min()
    for i_split in sorted(best_validations.keys()):
        # Split data into train/validation/test
        train_dt, valid_dt, test_dt = comm.datetime_splits(df, i_split,
                                                           config['n_splits'],
                                                           config['train_prop'])
        _, df_valid, df_test = comm.df_splits(df, i_split, train_dt,
                                              valid_dt, test_dt)
        _, ues_df_valid, ues_df_test = comm.df_splits(ues_df, i_split, train_dt,
                                                      valid_dt, test_dt)
        # Run test
        val_results = best_validations[i_split]['val_results']
        logging.info(f'Running test on best validation for split {i_split}.')
        test_costs = eval.run_evaluation(val_results['dueling_dnnetwork'],
                                         config['mitigation_cost'],
                                         i_split,
                                         df_test,
                                         ues_df_test,
                                         dt_min,
                                         valid_dt,
                                         test_dt,
                                         config['job_samples_path'],
                                         is_validation=False,
                                         df_valid=df_valid,
                                         ues_df_valid=ues_df_valid)
        test_costs.update({
            'i_split': i_split,
            'train_time': val_results['train_time'],
            'dueling_dnnetwork': val_results['dueling_dnnetwork'],
        })
        test_results.append(test_costs)

    # Summarize cross-validation results
    avg_ue_cost = sum([x['avg_ue_cost'] for x in test_results])
    avg_mitigation_cost = sum([x['avg_mitigation_cost'] for x in test_results])
    avg_cost = avg_ue_cost + avg_mitigation_cost
    logging.info('Average cross-validation (test) results:')
    logging.info(f'\tAvg. UE cost: {avg_ue_cost}')
    logging.info(f'\tAvg. mitigation cost: {avg_mitigation_cost}')
    logging.info(f'\tAvg. total cost: {avg_cost}')

    # Save final cross-validation results
    avg_ue_cost_split = [x['avg_ue_cost'] for x in test_results]
    avg_mitigation_cost_split = [x['avg_mitigation_cost'] for x in test_results]
    avg_cost_split = list(np.array(avg_ue_cost_split) + \
                          np.array(avg_mitigation_cost_split))
    cross_val_results = {
        'test_results': test_results,
        'avg_ue_cost_per_split': avg_ue_cost_split,
        'avg_mitigation_cost_per_split': avg_mitigation_cost_split,
        'avg_cost_per_split': avg_cost_split,
        'total_avg_ue_cost': avg_ue_cost,
        'total_avg_mitigation_cost': avg_mitigation_cost,
        'total_avg_cost': avg_cost
    }

    # Store test results
    comm.save_evaluation_results(cross_val_results,
                                 args['output'],
                                 eval_label='test')


if __name__ == "__main__":
    dt0 = dt.datetime.now()
    main()
    logging.info('Job ends running.')
    logging.info(f'Execution time: {dt.datetime.now() - dt0}')