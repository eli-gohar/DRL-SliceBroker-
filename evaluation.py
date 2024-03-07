import os
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_final_policy(n_eval_episodes, agent, env):
    print("\n----------------\nevaluate_final_policy()\n----------------\n")

    """ Evaluates the final policy of an agent for the specified experiment.
    Args:
      n_eval_episodes (int): number of evaluation episodes
      agent: The trained agent
      env: The environment the agent acts in
    """

    episode_results = dict()

    for episode_num in range(n_eval_episodes):
        episode = dict()
        rew, _ = evaluate_policy(agent, env, n_eval_episodes=1, return_episode_rewards=True)

        episode['total_nsr'] = env.total_nsr
        episode['reward'] = round(rew[0], 4)

        if (env.num_accepted + env.num_rejected) == 0:
            episode['acceptance_rate_local'] = env.num_accepted / 1
        else:
            episode['acceptance_rate_local'] = env.num_accepted / \
                                               (env.num_accepted + env.num_rejected)

        if env.num_accepted == 0 or env.total_nsr == 0:
            if env.num_accepted > -1 and env.total_nsr > -1:
                episode['acceptance_rate_global'] = env.num_accepted / 1
        else:
            episode['acceptance_rate_global'] = env.num_accepted / env.total_nsr

        episode['episode_length'] = env.episode_length
        episode['resource_costs'] = env.resource_costs['resource_costs']

        for key in env.resource_utilization:
            episode['{}'.format(key)] = env.resource_utilization[key]
        episode['placements'] = env.placements
        episode['operating_servers'] = env.operating_servers
        episode['cp_configurations'] = env.cp_configurations
        episode_results[episode_num] = episode

    return episode_results


def safe_experiment(results, args):
    """ Safes one experiment together with its metainformation into a csv file."""

    # Create output dir & save graphs
    os.makedirs(args['output'], exist_ok=True)

    # safe environment & hyper parameters used to generate the results
    with open(Path(args['output']) / 'args.json', 'a') as file:
        file.write(json.dumps(args))
        file.write("\n")

    # write determined embeddings to file
    table = []
    for ep, episode in results.items():
        for trial, logs in episode.items():
            decisions = logs.pop('placements')

            for sfc, embedding in decisions.items():
                row = [ep, trial, sfc.arrival_time, sfc.ttl, sfc.bandwidth_demand, sfc.max_response_latency, sfc.vnfs,
                       embedding]
                table.append(row)

    headers = ['Episode', 'Trial', 'Arrival', 'TTL', 'Bandwidth', 'Max Latency', 'VNFs', 'Placements']
    table = tabulate(table, headers=headers)
    with open(Path(args['output']) / 'placements.txt', 'w') as file:
        file.write(table)

    # safe agent's performances in csv format
    data = {(args['agent'], i, j): results[i][j] for i in results.keys()
            for j in results[i].keys()}
    print("safe_experiment: \n", data)
    # input()
    data = pd.DataFrame.from_dict(data, orient='index')
    data.index.names = ['agent', 'trial', 'episode']

    results_path = Path(args['output']) / 'results.csv'

    if not os.path.exists(results_path):
        data.to_csv(results_path)

    else:
        data.to_csv(results_path, mode='a', header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()


    def createDirectorIfNotExists(dirname):
        isExist = os.path.exists(dirname)
        if not isExist:
            os.mkdir(dirname)


    createDirectorIfNotExists(args.output)

    index_mapping = {'agent': 'Agent', 'trial': 'Trial', 'episode': 'Episode'}

    measure_mapping = {'total_nsr': 'Total NSRs',
                       'reward': 'Reward [0.0, inf]',
                       'acceptance_rate_local': 'Acceptance Rate (Local) [0.0, 1.0]',
                       'acceptance_rate_global': 'Acceptance Rate (Global) [0.0, 1.0]',
                       'episode_length': 'Episode Length [Sr]',
                       'resource_costs': 'Resource costs [Euro]',
                       'cpu_utilization': 'CPU Utilization [No. of vCPU] ',
                       'memory_utilization': 'Storage Utilization [GiB]',
                       'bandwidth_utilization': 'Bandwidth Utilization [Gbs]',
                       'icr_utilization': 'ICR Utilization [Gbs]',
                       'operating_servers': 'Operating CP [No. of CP]'}

    results = pd.DataFrame()

    for table_filename in ['results.csv']:
        table_data = pd.read_csv(Path(args.results) / table_filename)
        results = pd.concat((results, table_data))

    results = results.rename(columns={**index_mapping, **measure_mapping})
    results = results.replace('FirstFit_3', 'FirstFit')
    results = results.groupby(['Agent', 'Trial']).mean()
    results = results.reset_index()

    sns.set_style("whitegrid")
    for measure in measure_mapping.values():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Agent', y=measure, data=results, ax=ax)
        sns.despine()
        fig.savefig(Path(args.output) / f'{measure}.svg')
