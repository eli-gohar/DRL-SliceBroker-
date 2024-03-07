import os
import json
import logging
import argparse
from pathlib import Path
from tensorboard import program  # for real-time TensorBoard integration
from stable_baselines3 import A2C, PPO
from evaluation import evaluate_final_policy, safe_experiment
from src.environment.inp_topology import createGraph2
from src.environment.env import Env
from src.agent.baselines import *
from src.agent.logging import MetricLoggingCallback, CustomMonitor

# Set environment variable to avoid OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_agent_type_and_policy(agent_name):
    agents_map = {
        'Random': (BaselineHeuristic, RandomPolicy),
        'FirstFit_1': (BaselineHeuristic, FirstFitPolicy),
        'FirstFit_2': (BaselineHeuristic, FirstFitPolicy2),
        'FirstFit_3': (BaselineHeuristic, FirstFitPolicy3),
        'FirstFit_4': (BaselineHeuristic, FirstFitPolicy4),
        'A2C': (A2C, 'MlpPolicy'),
        'PPO': (PPO, 'MlpPolicy'),
    }
    if agent_name not in agents_map:
        raise ValueError('An unknown agent was specified')
    return agents_map[agent_name]


def enable_real_time_tensor_board(enable, tracking_address, agent_name=None):
    if enable:
        # Launch TensorBoard
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tracking_address])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")
    else:
        print("Real-time TensorBoard inactive")

    if agent_name:
        # Find the latest log directory
        latest_log_dir = find_latest_log_dir(tracking_address, agent_name)
        if latest_log_dir:
            print("Check the following folder to view TensorBoard logs:", latest_log_dir)



def find_latest_log_dir(tracking_address, agent_name):
    import re

    # Get the list of all directories in tracking_address
    dir_list = os.listdir(tracking_address)

    def extract_number(f):
        s = re.findall(r'\d+$', f)
        return int(s[0]) if s else -1

    dir_numbers = [extract_number(d) for d in dir_list]
    max_dir_number = max(dir_numbers, default=-1)

    if max_dir_number >= 0:
        next_dir_number = max_dir_number + 1
        latest_log_dir = os.path.join(tracking_address, f"{agent_name}_{next_dir_number}")
        return latest_log_dir

    return None


if __name__ == '__main__':

    # Parse command-line arguments

    parser = argparse.ArgumentParser()

    # arguments to specify parameters of the experiment evaluation

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RL Agent Experiment')
    parser.add_argument('--agent', type=str, required=True, help='Type of RL agent or baseline')
    parser.add_argument('--total_train_timesteps', type=int, nargs='?', const=1, default=1048576,
                        help='Number of training steps for the agent (better if in multiple of 2)')
    parser.add_argument('--logs', type=str, nargs='?', const=1, default='./logs', help='Path of tensorboard logs')
    parser.add_argument('--tensorboard', type=bool, nargs='?', const=1, default=False,
                        help='Enable Real Time TensorBoard')

    # ... (other command-line arguments related to the experiment)
    # arguments to specify the InP network and NSRs
    parser.add_argument('--network', type=str, required=True,
                        help='Path to network graph for the environment')
    parser.add_argument('--requests', type=str,
                        help='Either path to request file or config of stochastic arrival process')

    # arguments to specify the final policy's evaluation
    parser.add_argument('--eval_episodes', type=int,
                        default=1, help='Number of evaluation steps for one trained agent')
    parser.add_argument('--trials', type=int, default=2,
                        help='Number of trials evaluating the agent')
    parser.add_argument('--save_model', type=str, default='./4_model/',
                        help='Path to the folder where all results will be stored at')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the folder where all results will be stored at')

    # arguments to specify debugging the logs of the environment
    parser.add_argument('--debug', action='store_false',
                        help='Whether to enable debugging logs of the environment')
    args = parser.parse_args()

    # Create agent from experiment configuration
    agent_name = args.agent
    agent_type, policy = get_agent_type_and_policy(agent_name)

    # Set logging level according to --debug
    logging.basicConfig()
    debug_level = logging.INFO if args.debug else logging.DEBUG
    logging.getLogger().setLevel(debug_level)

    # Create log dir & monitor training so that episode rewards are logged
    os.makedirs(args.logs, exist_ok=True)

    # load the NSR arrival process properties
    with open(Path(args.requests), 'r') as file:
        arrival_config = json.load(file)

    createGraph2(args.network, num_mec=4, num_dc=12, graph_seed=1, show_graph=False)

    # Parse the network from gpickle
    if isinstance(args.network, str) and args.network.endswith('.gpickle'):
        network = nx.read_gpickle(args.network)

    # enable_real_time_tensor_board(True, args.logs)
    enable_real_time_tensor_board(False, args.logs)

    results = dict()
    for trial in range(args.trials):

        # create the network structure & incoming requests for the environment
        arrival_config['seed'] = trial
        env = Env(args.network, arrival_config)

        env = CustomMonitor(env, args.logs)

        callback = MetricLoggingCallback()

        agent = agent_type(policy=policy, env=env, verbose=1, tensorboard_log=args.logs)
        tb_log_name = agent.__class__.__name__ if isinstance(policy, str) else policy.__name__

        if policy == 'MlpPolicy':
            # Only MLP policies require training
            agent.learn(total_timesteps=args.total_train_timesteps, tb_log_name=tb_log_name, callback=callback)
            agent.save(args.save_model + agent_name + "_" + str(trial))

        # Evaluate the final policy and log performances
        results[trial] = evaluate_final_policy(args.eval_episodes, agent, env)

    # save experiments to disk at specified output path
    safe_experiment(results, vars(args))
