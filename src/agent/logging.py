from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from tabulate import tabulate


def pause(bool):
    if bool:
        input()
        input()


class MetricLoggingCallback(BaseCallback):
    """Custom callback that logs the agent's performance to TensorBoard.
        More on https://www.tensorflow.org/guide/keras/custom_callback
    """

    def __init__(self, verbose=0):
        super(MetricLoggingCallback, self).__init__(verbose)
        """
        The super() function is used to give access to methods and properties of a parent or sibling class.
        """

    def _on_step(self):
        """Logs step information of custom environment to TensorBoard"""
        ## get monitor object and log network metrics to TensorBoard
        ## Supports console output True/False

        # display monitor to console output?
        # monitor_console_output = True
        # pause_console_output = True

        monitor_console_output = False
        pause_console_output = False

        # get monitor object and log network metrics to TensorBoard
        monitor: CustomMonitor = self.training_env.envs[0]
        # print("monitor.num_accepted: ", monitor.num_accepted)
        def GetProcessingRequestNumber():
            num_requests = max(monitor.num_accepted + monitor.num_rejected, 1)
            if monitor.num_accepted + monitor.num_rejected == 0:
                num_requests = 1
            if monitor.num_accepted or monitor.num_rejected > 0:
                num_requests = (monitor.num_accepted + monitor.num_rejected)
                num_requests += 1
            return num_requests


        if monitor.total_nsr != 0 and monitor.total_nsr > 0:
            if monitor.num_accepted != 0 and monitor.num_accepted > 0:
                self.logger.record('1 - Ratio/acceptance_ratio_global',
                                   monitor.num_accepted / monitor.total_nsr)

        num_requests = GetProcessingRequestNumber()

        # log sfc related information for the respective episode
        self.logger.record('1 - Ratio/acceptance_ratio_local',
                           monitor.num_accepted / num_requests)

        # print(num_requests, monitor.num_accepted, monitor.num_rejected)

        self.logger.record('1 - Ratio/rejection_ratio',
                           monitor.num_rejected / num_requests)

        # log total costs for the respective episode
        self.logger.record('2 - total_costs', sum(monitor.resource_costs.values()))

        # log the amount of occupied resources per resource type
        ## for key in monitor.resource_utilization: ## loop ignores a key if value is None. Manually assign values :)
        ##     self.logger.record('{}'.format(key), monitor.resource_utilization[key])
        self.logger.record('6 - Utilization/cpu_utilization', monitor.resource_utilization['cpu_utilization'])
        self.logger.record('6 - Utilization/memory_utilization', monitor.resource_utilization['memory_utilization'])
        self.logger.record('6 - Utilization/icr_utilization', monitor.resource_utilization['icr_utilization'])
        self.logger.record('6 - Utilization/bandwidth_utilization',
                           monitor.resource_utilization['bandwidth_utilization'])



        # log the number of operating CPs per step in an episode
        self.logger.record('4 - operating_servers', monitor.operating_servers)

        # log the running configuration on each CP
        if monitor.cp_configurations:  ## if dict is not empty
            for key in monitor.cp_configurations:  ## get the key of dict
                val = monitor.cp_configurations.get(key)  ## check if key exists in the dict
                if val is not None:
                    self.logger.record('5 - Config of CP/CP {}'.format(key), val)
                else:
                    self.logger.record('5 - Config of CP/CP {}'.format(key), -1)

        # log the reward of an agent
        self.logger.record('3 - reward', monitor.reward)

        if monitor_console_output:
            table_data = [monitor.total_nsr, monitor.num_accepted, monitor.num_rejected, monitor.episode_length,
                          sum(monitor.resource_costs.values()),
                          round(monitor.resource_utilization['cpu_utilization'], 2),
                          round(monitor.resource_utilization['memory_utilization'], 2),
                          round(monitor.resource_utilization['icr_utilization'], 2),
                          round(monitor.resource_utilization['bandwidth_utilization'], 2),
                          monitor.operating_servers, monitor.cp_configurations]

            table_header = ["total_nsr", "num_req accepted", "num_req rejected", "episode_length",
                            "resource_costs",
                            "cpu_utilization", "memory_utilization", "icr_utilization", "bandwidth_utilization",
                            "operating_servers", "cp_configurations"]
            tensor_board_data = [table_header, table_data]

            print("\n----------------\nTensorBoard()\n----------------")
            print("proc. request num", num_requests)
            print("reward: ", round(monitor.reward, 2))
            print(tabulate(tensor_board_data, headers="firstrow", tablefmt="git"))

        pause(pause_console_output)


class CustomMonitor(Monitor):
    """Custom monitor tracking additional metrics that ensures compatability with StableBaselines."""

    def reset(self, **kwargs):
        """Augments the environment's monitor with network related metrics."""
        self.total_nsr = 0
        self.num_accepted = 0
        self.num_rejected = 0
        self.episode_length = 1
        self.resource_costs = {'resource_costs': 0.0}
        self.resource_utilization = {'cpu_utilization': 0.0,
                                     'memory_utilization': 0.0,
                                     'bandwidth_utilization': 0.0,
                                     'icr_utilization': 0.0}

        self.operating_servers = 0

        self.cp_configurations = {}

        self.placements = {}

        return super().reset(**kwargs)

    def step(self, action):
        """Extract the environment's information i.e., step information, to the monitor."""
        ## get step information from environment and log network metrics to monitor
        ## Supports console output True/False

        # display step to console output?
        # step_console_output = True
        # pause_console_output = False

        step_console_output = False
        pause_console_output = False

        # print("Extract the environment's information i.e., step information, to the monitor")
        observation, reward, done, info = super(CustomMonitor, self).step(action)

        # print("Environment's information extracted to the monitor.")

        # add sfc related information to the monitor

        self.num_accepted += info['accepted']
        self.num_rejected += info['rejected']

        if info['accepted'] or info['rejected']:
            sfc = info['sfc']
            self.placements[sfc] = info['placements']

        # add resource cost related information to the monitor
        try:
            self.total_nsr = info['total_nsr']
            self.resource_costs['resource_costs'] = info['resource_costs']
        except:
            KeyError

        try:
            # log the amount of occupied resources per resource type
            self.resource_utilization['cpu_utilization'] = info['cpu_utilization']
            self.resource_utilization['memory_utilization'] = info['memory_utilization']
            self.resource_utilization['icr_utilization'] = info['icr_utilization']
            self.resource_utilization['bandwidth_utilization'] = info['bandwidth_utilization']
        except:
            KeyError

        try:
            # log the number of operating CPs per step in an episode
            self.operating_servers = info['operating_servers']
            # log the configurations of each CP
            self.cp_configurations = info['cp_configurations']

        except:
            KeyError

        if step_console_output:
            step_table_data = [self.total_nsr, self.num_accepted, self.num_rejected, self.episode_length,
                               self.resource_costs['resource_costs'],
                               round(self.resource_utilization['cpu_utilization'], 2),
                               round(self.resource_utilization['memory_utilization'], 2),
                               round(self.resource_utilization['icr_utilization'], 2),
                               round(self.resource_utilization['bandwidth_utilization'], 2),
                               self.operating_servers, self.cp_configurations]

            self.episode_length += 1

            step_table_header = ["total_nsr", "num_req accepted", "num_req rejected", "episode_length",
                                 "resource_costs",
                                 "cpu_utilization", "memory_utilization", "icr_utilization", "bandwidth_utilization",
                                 "operating_servers", "cp_configurations"]
            step_data = [step_table_header, step_table_data]

            print("\n----------------\nmonitor()\n----------------")
            print("reward", round(reward, 2))
            print(tabulate(step_data, headers="firstrow", tablefmt="git"))

        pause(pause_console_output)

        return observation, reward, done, info
