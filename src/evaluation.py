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

from . import job_funs


def get_mitigation_cost(finished_mitis_per_node, miti_cost):
    """
    Get the total mitigation cost in node-hours.

    Args:
        finished_mitis_per_node (pandas.Series): Series with id_blade as
                                                 keys. Each list element
                                                 is the finished
                                                 mitigation datetime.
                                                 It is an empty list if 
                                                 there is no mitigation 
                                                 for that node.
        miti_cost (dt.timedelta): Cost of a mitigation action.

    Returns:
        float: Total mitigation cost in node-hours.
    """
    if finished_mitis_per_node.empty:
        return 0
    total_miti_cost = (finished_mitis_per_node.apply(len).sum() * miti_cost)
    if total_miti_cost == 0:
        # Special check because when it is 0, then it is not a timedelta
        # so we cannot check `total_seconds()`
        return 0
    return total_miti_cost.total_seconds() / 3600


def get_ue_cost(ues_df, finished_mitis_per_node, jobs_distr):
    """
    Get the total UE cost in node-hours.

    Args:
        ues_df (pandas.DataFrame): UEs dataframe.
        finished_mitis_per_node (pandas.Series): Series with id_blade as
                                                 keys. Each list element
                                                 is the finished
                                                 mitigation datetime.
                                                 It is an empty list if 
                                                 there is no mitigation 
                                                 for that node.
        jobs_distr (dict): Jobs distribution.
    """
    ue_costs = []
    for ue in ues_df.itertuples():
        ue_blade = ue[-1]
        ue_dt = ue[4]
        ue_job_hours = 0

        # Get the last job that started on the node
        starts = pd.to_datetime(jobs_distr[ue_blade]['start'])
        last_job_start_idx = np.where(starts <= ue_dt)[0][-1]
        if jobs_distr[ue_blade]['hours_cost'][last_job_start_idx] == 0:
            # Continue if the node is idle
            ue_costs.append(ue_job_hours)
            continue

        # Compute UE cost
        last_job_start = starts[last_job_start_idx]
        last_job_n_nodes = jobs_distr[ue_blade]['nnodes'][last_job_start_idx]
        secs_since_job_start = (ue_dt - last_job_start).total_seconds()
        job_size = secs_since_job_start * last_job_n_nodes
        ue_job_hours = job_size / 3600

        # Check if there were any mitigations
        if ue_blade not in finished_mitis_per_node or \
           len(finished_mitis_per_node[ue_blade]) == 0:
            # Continue if the node is not mitigated
            ue_costs.append(ue_job_hours)
            continue
        finished_mitis_node = pd.to_datetime(finished_mitis_per_node[ue_blade])
        is_prev_miti = finished_mitis_node < ue_dt
        if len(finished_mitis_node[is_prev_miti]) == 0:
            # Continue if there is no previous mitigation before the UE
            ue_costs.append(ue_job_hours)
            continue

        # Consider time since last mitigation
        finished_dt = finished_mitis_node[is_prev_miti][-1]
        secs_since_miti = (ue_dt - finished_dt).total_seconds()
        secs_cost = secs_since_miti * last_job_n_nodes
        hours_since_miti_cost = secs_cost / 3600
        
        # Add total UE cost, which is the minimum between hours since
        # the last job started or hours since the last mitigation
        ue_costs.append(min(ue_job_hours, hours_since_miti_cost))
    
    return ue_costs


def get_agent_finished_mitigations(blade_group, blade_jobs_distr,
                                   action_fn, mitigation_cost):
    """
    Get the datetimes of finished mitigations per blade/node.

    Args:
        blade_group (pandas.DataFrame): Group of events per blade.
        blade_jobs_distr (dict): Blade jobs distribution.
        action_fn (function): Action function.
        mitigation_cost (dt.timedelta): Cost of a mitigation action.

    Returns:
        pandas.Series: Series with id_blade as keys. Each list element 
                       is the finished mitigation datetime.
    """
    fin_mitis = []
    starts = pd.to_datetime(blade_jobs_distr['start'])
    # Loop through each set of events (sequentially by date)
    for row in blade_group.itertuples():
        curr_dt = row[1]
        last_job_start_idx = np.where(starts <= curr_dt)[0][-1]
        potential_cost = 0
        if blade_jobs_distr['hours_cost'][last_job_start_idx] != 0:
            # If the node is not idle, add penalization to the reward value
            # Get potential UE cost in case of no mitigation
            last_job_start = starts[last_job_start_idx]
            last_job_n_nodes = blade_jobs_distr['nnodes'][last_job_start_idx]
            secs_since_job_start = (curr_dt - last_job_start).total_seconds()
            secs_cost = secs_since_job_start * last_job_n_nodes
            ue_job_hours = secs_cost / 3600
            # Get cost since last mitigation
            hours_since_mitigation = np.inf
            if len(fin_mitis):
                # If there are any mitigations
                secs_since_last_miti = (curr_dt - fin_mitis[-1]).total_seconds()
                secs_cost = secs_since_last_miti * last_job_n_nodes
                hours_since_mitigation = secs_cost / 3600
            # Potential cost is equal to the minimum
            potential_cost = min(ue_job_hours, hours_since_mitigation)
        # Agent mitigates or not based on features and potential cost
        action = action_fn(np.append(row[3:], potential_cost))
        # Save time of last mitigation finish
        if action == 1:
            fin_mitis.append(curr_dt + mitigation_cost)
    
    return np.array(fin_mitis)


def eval_strategy(df_split, ues_df_split, finished_mitigations,
                  job_samples_path, mitigation_cost,
                  decision_depends_on_job_distribution=False):
    """
    Evaluate a given strategy known by finished mitigations. Finished 
    mitigations can be a callable function or a list of mitigations for 
    each job sample.

    Args:
        df_split (pd.DataFrame): Split features dataframe.
        ues_df_split (pd.DataFrame): Split UEs dataframe.
        finished_mitigations (callable or list): Finished mitigation 
                                                 datetimes.
        job_samples_path (str): Path to job samples.
        mitigation_cost (dt.timedelta): Cost of a mitigation action.
        decision_depends_on_job_distribution (bool, optional): Whether 
            the mitigation decision depends on job distribution.
            Defaults to False.

    Returns:
        dict: Evaluation results -->
            ue_costs (list): Total UE cost per job sample,
            ue_costs_lists (list): All UE costs per job sample,
            miti_costs (list): Mitigation costs per job sample,
            finished_mitis (pd.Series): Finished mitigations per node
    """
    miti_costs = []
    ue_costs_list = []
    ue_costs = []

    if callable(finished_mitigations):
        # Compute finished mitigations from the given function
        finished_mitis = finished_mitigations(df_split)
    else:
        finished_mitis = finished_mitigations

    if not decision_depends_on_job_distribution:
        # Optimization: Compute total mitigation here if the mitigation 
        # decision does not depend on job distribution
        mitic = get_mitigation_cost(finished_mitis, mitigation_cost)

    for i, job_file in enumerate(os.listdir(job_samples_path)):
        sample_path = os.path.join(job_samples_path, job_file)
        jobs_distr = job_funs.get_job_distribution(sample_path)
        if decision_depends_on_job_distribution:
            # Compute total mitigation cost for each job sample
            fin_mitis_in_sample = finished_mitis[i]
            mitic = get_mitigation_cost(fin_mitis_in_sample, mitigation_cost)
        else:
            fin_mitis_in_sample = finished_mitis
        # Compute total UE cost
        uec = get_ue_cost(ues_df_split, fin_mitis_in_sample,
                          jobs_distr)
        ue_costs_list.append(uec)
        ue_costs.append(sum(uec))
        miti_costs.append(mitic)
    
    return {
        'ue_costs': np.array(ue_costs),
        'ue_costs_lists': np.array(ue_costs_list),
        'mitigation_costs': np.array(miti_costs),
        'finished_mitigations': finished_mitis,
    }


def evaluate(df_split, ues_df_split, model, job_samples_path, mitigation_cost):
    """
    Evaluate the given model (normally an RL agent) on the given dataframe.

    Args:
        df_split (pd.DataFrame): Split features dataframe.
        ues_df_split (pd.DataFrame): Split UEs dataframe.
        model (RL agent): RL agent.
        job_samples_path (str): Path to job samples.
        mitigation_cost (dt.timedelta): Cost of a mitigation action.

    Returns:
        dict: Evaluation results -->
            ue_costs (list): Total UE cost per job sample,
            miti_costs (list): Mitigation costs per job sample,
            finished_mitigations (list): Finished mitigations in all job samples
    """
    # Set action function based on the model's decision
    action_fn = lambda obs: model.get_action(obs, epsilon=0)

    # Finished mitigations in all job samples
    all_fin_mitis = []
    for job_file in os.listdir(job_samples_path):
        # Get all the mitigations performed at each node by the agent 
        # based on the given jobs sample
        sample_path = os.path.join(job_samples_path, job_file)
        jobs_distr = job_funs.get_job_distribution(sample_path)
        agent_finished_mitis = df_split.groupby('id_blade').apply(
            lambda g: get_agent_finished_mitigations(
                g, jobs_distr[g.iloc[0]['id_blade']], action_fn,
                mitigation_cost))
        all_fin_mitis.append(agent_finished_mitis)

    ev_agent = eval_strategy(df_split, ues_df_split, all_fin_mitis,
                             job_samples_path, mitigation_cost,
                             decision_depends_on_job_distribution=True)

    return {
        'ue_costs': ev_agent['ue_costs'],
        'mitigation_costs': ev_agent['mitigation_costs'],
        'avg_ue_cost': np.mean(ev_agent['ue_costs']),
        'avg_mitigation_cost': np.mean(ev_agent['mitigation_costs']),
        # 'finished_mitigations': all_fin_mitis,
    }


def run_evaluation(dueling_dnnetwork, mitigation_cost, i_split, df2eval,
                   ues2eval, dt_min, start_dt, end_dt, job_samples_path,
                   is_validation=False, df_valid=None, ues_df_valid=None):
    """
    Evaluate RL agent.

    Args:
        agent (RL agent): RL agent.
        mitigation_cost (dt.timedelta): Cost of a mitigation action.
        i_split (int): Number specifying the ith split to be used for
                       training/validation/testing.
        df2eval (pd.DataFrame): Features to evaluation (train/validatoin/test).
        ues2eval (pd.DataFrame): UEs used for evaluation (train/validatoin/test).
        dt_min (dt.datetime): Minimum datetime in features dataframe.
        start_dt (dt.datetime): Datetime when starting evaluation.
        end_dt (dt.datetime): Datetime when ending evaluation.
        job_samples_path (str): Path to job samples directory.
        is_validation (bool, optional): Flag for validation or testing.
        df_valid (pd.DataFrame, optional): Validation data, required only if
                                           is_validation.
        ues_df_valid (pd.DataFrame, optional): Validation UEs data, required
                                               only if is_validation.

    Returns:
        dict: Evaluation results.
    """
    # Evaluate RL agent
    eval_costs = evaluate(df2eval.reset_index(), ues2eval, dueling_dnnetwork,
                          job_samples_path, mitigation_cost)
    # Add extra info to evaluation costs
    eval_costs.update({
        'start_dt': start_dt,
        'end_dt': end_dt,
        'is_validation': is_validation,
    })

    if i_split == 0 and not is_validation:
        # If testing first split, add never-mitigate costs from the minimum
        # datetime until valid_dt, this is because in the first weeks we
        # don't have any trained model
        no_mitigate_fn = lambda filtered_df: filtered_df.groupby('id_blade')\
                                                        .apply(lambda _: [])
        no_mitigate_costs = eval_strategy(
            df_valid, ues_df_valid, no_mitigate_fn, job_samples_path,
            mitigation_cost, decision_depends_on_job_distribution=False)
        # As we're taking into account the whole split for testing,
        # consider the start as the first datetime.
        eval_costs['start_dt'] = dt_min
        # Sum all job_range UE costs for considering a no-mitigation
        # strategy during the validation phase.
        # There is no need to modify mitigation costs and finished
        # mitigations, as there are no mitigations during this period.
        eval_costs['ue_costs'] += no_mitigate_costs['ue_costs']
        # Update averages
        eval_costs['avg_ue_cost'] = eval_costs['ue_costs'].mean()

    return eval_costs