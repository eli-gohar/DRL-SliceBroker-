import os

import gym
import logging
import copy
import operator
import functools
import collections
import numpy as np
import pandas as pd
from gym import spaces
from copy import deepcopy
from tabulate import tabulate
from src.environment.network import Network
from src.environment.nsr import runRequestCreation, readRequestsFromFileForInput


class Env(gym.Env):
    def __init__(self, overlay_path, arrival_config):
        """A new Gym environment. This environment represents the environment of the VNF allocaion.

        Args:
          overlay_path: A connection to the network where the agent will act
          arrival_config: Dictionary that specifies the properties of the requests."""

        self.overlay_path = overlay_path
        self.arrival_config = arrival_config

        ## define action and state space
        _, properties = Network.check_overlay(self.overlay_path)
        self.num_nodes = properties['num_nodes']
        num_node_resources = properties['num_node_resources']

        num_node_resources -= 1

        """
        why "num_node_resources -= 1"?
        The node resources that are part of an observation are: 'cpu', 'memory', 'bandwidth', 'icr', and 'icd'
        The node 'type' is not part of an observation. Therefore, we remove the 'type' by decrementing num_node_resources 
        """
        """
        Environment:
        We build a customized OpenAI Gym (Gym) environment for simulation
        The enviornment is Vectorized and uses DummyVecEnv 
        We use a monitor wrapper for the Gym environment, it is used to know the episode reward, length, time and other data.
        
        Spaces:
        The observation_space for our environment is Box(), and 
        The action_space for our environment is Discrete()
        """

        # action space
        self.action_space = spaces.Discrete(self.num_nodes)

        # observation space
        obs_dim = self.num_nodes * num_node_resources + num_node_resources + 1

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.reward = 0

        self.count_not_allocated = 0

        ## To inform the system that "previous_request is completed"
        self.pre_batch_completed = False

        ## To inform the system that current SFC VNF being processed is valid which means
        ## Constraints such CPU, STO, ICR and bandwidth, are satisfied by VNF and
        ## Constraints such as end-to-end is satisfied by SFC.
        self.is_valid_sfc = False

        ## To inform the system that current VNF being processed is last VNF of SFC
        self.is_last_of_sfc = False

        self.is_sfc_rejected = False

        ### Experimental:
        self.seed_sample = 0

        # initialize arrival process that will generate SFC requests
        self.request_file = runRequestCreation()

    def reset(self):
        """
            Resets the environment to the default state.

            Returns:
            - observation (numpy.ndarray): The initial state representation of the environment.
            """

        ## Reset flags and counters
        self.done = False
        ## total number of requests
        self.total_nsr = 0

        ## initialize 'network' and backtracking network which is 'vnf_backtrack'
        ## 'vnf_backtrack' is updated after each successful VNF allocation
        self.network = Network(self.overlay_path)
        self.vnf_backtrack = deepcopy(self.network)

        ## track the indices for current SFC & VNF
        self.sfc_idx, self.vnf_idx = 0, 0

        ## current SFC request i.e. list of SFC VNFs to be processed
        self.request_batch = []

        ## progress time until first batch of requests arrives
        self.progress_inter_timeslots()

        self.nsr_staistics = []  ## to store the statistics of
        self.record_network = {}

        ## Compute the environment's state representation and return observation
        ## When resetting the environment we call compute_state() with is_reset_called=True
        ###  When is_reset_called=True then the input state is all zeros to the neural network
        return self.compute_state(is_reset_called=True)

    def progress_inter_timeslots(self):
        """ Progress `request_batch` and `network` over inter timeslots, i.e. until novel requests arrive.
        """
        ### empty the 'request_batch' since we assume that all requests have been processed
        self.request_batch = []

        try:
            while len(self.request_batch) <= 0:  ## While the request_batch is empty
                time_step = self.network.update()  ## update system time and get time_step
                if time_step == 1:
                    ### read all requests from file at time_step=1 and save it in `arrival_process`
                    self.arrival_process, self.total_nsr = readRequestsFromFileForInput(self.request_file)

                ### get SFC request based on 'arrival_time' from 'self.arrival_process'
                self.request_batch = self.getNext(time_step)

                if self.request_batch == None:
                    ### the arrival_process is done generating further requests, finish the episode
                    self.request_batch = []
                    return True

                if self.request_batch:  ### Show if a SFC request is available
                    pass

            ### initialize backtracking for the invoked timeslot
            self.vnf_backtrack = deepcopy(self.network)
            return False

        except StopIteration:
            ### the `arrival_process` is done generating further requests, finish the episode
            return True

    ### Experimental:
    def deterministicAction(self, case):
        """Genereate deterministic action of the agent.
        Args:
            case(int):
        Return:
            action(int): The action is the number of the server where the current VNF will be embedded to.
        """
        ## For all simulations
        if case == 'alwaysOneActionValue':  ## It will always generate one action value when step() is called.
            self.action_space.seed(12)  ## in this simulation setting it will always generate 3
            return self.action_space.sample()
        elif case == 'alwaysSameActionValue':  ## It will always generate same action value when step() is called.
            self.seed_sample += 1
            self.action_space.seed(self.seed_sample)
            return self.action_space.sample()

    def step(self, action):
        """
        Process the action of the agent.

        Args:
            action (int): The action is the number of the CP where the current VNF will be embedded to.

        Returns:
            observation (numpy.ndarray): The state representation of the environment after the step.
            reward (float): The reward received after the step.
            done (bool): True if the episode is done, False otherwise.
            info (dict): Additional information about the step.
    """

        # Information dictionary to track the state of the NSRs
        info = {'accepted': False, 'rejected': False, 'total_nsr': self.total_nsr}

        # Ensure the episode is not already finished before processing the step
        assert (not self.done), 'Episode was already finished, but step() was called'

        if self.pre_batch_completed:
            ## if "previous_request is completed" then check if we can:
            self.done = self.progress_inter_timeslots()  ## get novel_request and episode information
            ## if we get novel_request then it is current_request and
            ## current_request is not completed therefore, previous_request is updated to False
            self.pre_batch_completed = False

            # Check if the episode is ends?
            if self.done:  # if episode ends i.e., self.done==True
                # Save statistics to a CSV file when the episode is finished
                pd_records = pd.DataFrame(self.nsr_staistics)
                isExist = os.path.exists('5_record')

                if isExist:
                    pd_records.to_csv('5_record\\global_summary.csv', index=False)
                else:
                    os.makedirs('5_record')
                    pd_records.to_csv('5_record\\global_summary.csv', index=False)

                # Reset the reward and return the current state
                self.reward = 0.0
                return self.compute_state(done=self.done), self.reward, self.done, info

        ### Experimental: To sample deterministic action
        ### comment it to sample agents real action
        # case = 'alwaysOneActionValue'  ## It will always generate one action value when step() is called.
        # case = 'alwaysSameActionValue'  ## It will always generate same action value when step() is called.
        # action = self.deterministicAction(case)

        ## get the indices for the to be processed SFC VNF
        sfc = self.request_batch[self.sfc_idx]
        vnf = sfc.vnfs[self.vnf_idx]

        '''
        Allocate the VNF to the CP. 
        An allocation is invalid if there are insufficient resources to allocate the current SFC's VNF or 
        if the SFC fails to meet its constraints, such as bandwidth and latency.
        '''

        is_valid_sfc = self.vnf_backtrack.embed_vnf(sfc, vnf, action)  ## Checks CPU, STO and ICR

        if is_valid_sfc:
            is_valid_sfc = self.vnf_backtrack.check_embeddable(sfc, action)  ## Checks bandwidth and latency

        '''
        Process the current sampled action.   
        If the action was invalid, return to the previous network state; 
        otherwise, proceed to the next state after a successful VNF allocation.
        '''

        is_last_of_sfc = self.vnf_idx >= len(sfc.vnfs) - 1
        ## determine whether the VNF being processed is the current SFC's last VNF?

        if is_valid_sfc and not is_last_of_sfc:
            pass

        elif is_valid_sfc and is_last_of_sfc:
            ## If current VNF being processed is last VNF of current SFC then
            ## the allocation of all VNFs of current SFC were allocated and
            ## we have successful SFC allocation
            pass

            ## Update network state to `vnf_backtrack` after successful SFC allocation
            self.network = deepcopy(self.vnf_backtrack)

            ## update info regarding successful SFC allocation
            info['accepted'] = True
            info['placements'] = self.vnf_backtrack.sfc_embedding[sfc]
            info['sfc'] = sfc
            # self.accepted = self.accepted + 1

        elif not is_valid_sfc:
            ## backtrack to the latest network state if the VNF allocation is invalid
            info['sfc'] = sfc  ## update info regarding unsuccessful SFC allocation
            if not self.vnf_backtrack.sfc_embedding.get(sfc):  ## check if SFC exists
                ## If first vnf of an sfc cannot be placed on CP then we do nothing
                info['placements'] = None
                self.count_not_allocated += 1
            else:
                ## If first or more vnf of an sfc has been placed on CP then we save their information
                info['placements'] = self.vnf_backtrack.sfc_embedding[sfc]  ## We keep previous sfc placement nodes
                self.count_not_allocated += 1

        ## We try to allocate current VNF being processed for (x) number of steps (no_of_steps)
        ## If the agent cannot allocate the current VNF being processed after (x) number of steps we drop the SFC
        no_of_steps = 5
        self.noValidAllocationFound(no_of_steps, info, sfc)

        ## All if cases end here

        self.reward = self.compute_reward(sfc, is_last_of_sfc, is_valid_sfc, batch_completed=True)

        self.is_valid_sfc = is_valid_sfc
        self.is_last_of_sfc = is_last_of_sfc
        self.compute_cost_reward(sfc, info)

        return self.compute_state(done=self.done), self.reward, self.done, info

    def compute_cost_reward(self, sfc, info):
        resource_utilization = self.vnf_backtrack.calculate_resource_utilization()

        ## sum the amount of utilized resources per resource type for all nodes
        sum_utilization = dict(functools.reduce(operator.add,
                                                map(collections.Counter,
                                                    resource_utilization)))

        if sum_utilization:
            ## Round Off Dictionary Values to 2 decimals
            sum_utilization = {key: sum_utilization[key] for key in sum_utilization}
        else:
            ## if none of the nodes were utilized then we initialize all node attributes to zero
            sum_utilization = {'cpu': 0.0, 'memory': 0.0, 'bandwidth': 0.0, 'icr': 0.0}

        resource_costs, cp_configurations, self.record_network = self.vnf_backtrack.calculate_resource_costs(sfc)

        ## log the occupied resources and operation costs, log the number of operating nodes
        # info.update({res + '_utilization': val for res, val in sum_utilization.items()}) ## do not round dic values
        info.update({res + '_utilization': round(val, 2) for res, val in sum_utilization.items()})  ## round dic values

        if 'icr' not in sum_utilization:
            info.update({'icr_utilization': 0.0})

        info.update({res + '_costs': val for res, val in resource_costs.items()})
        num_operating = len(self.vnf_backtrack.get_operating_nodes())
        info.update({'operating_servers': num_operating})
        info.update({'cp_configurations': cp_configurations})

        # if self.is_valid_sfc and self.is_last_of_sfc:
        ## self.reward is 0 when VNF is not placed by an agent
        if info['resource_costs'] != 0 and self.reward != 0:
            self.reward += (1 / info['resource_costs'])
            info['resource_costs'] = round(info['resource_costs'], 7)
        else:
            self.reward += 0.0

        if self.is_valid_sfc and self.is_last_of_sfc:
            self.recordSfc(sfc, description="Success", place_result=True, route_result=True)
            self.recordInp(info)
            self.nsr_staistics.append(copy.deepcopy(self.record_network))

    def printInfo(self, info, info_sfc=False):

        table_data = []
        table_data.append(info['accepted'])
        table_data.append(info['rejected'])
        table_data.append(self.reward)
        table_data.append(info['resource_costs'])

        table_data.append(info['cpu_utilization'])
        table_data.append(info['memory_utilization'])

        table_data.append(info['icr_utilization'])
        table_data.append(info['bandwidth_utilization'])

        table_data.append(info['operating_servers'])
        table_data.append(info['cp_configurations'])

        table_header = ["curr_req accepted", "curr_req rejected", "reward", "resource_costs",
                        "cpu_utilization", "memory_utilization", "icr_utilization", "bandwidth_utilization",
                        "operating_servers", "cp_configurations"]
        monitor_data = [table_header, table_data]

    def noValidAllocationFound(self, no_of_steps, info, sfc):
        """
              Handle the case where no valid allocation is found after a certain number of steps.

              Parameters:
                  - no_of_steps (int): The threshold number of steps after which no valid allocation is considered.
                  - info (dict): A dictionary containing information about the allocation process.
                  - sfc (object): An object representing the Service Function Chain.

              Returns:
                  None

              Actions:
                  - Checks if the count of unsuccessful allocations is equal to the specified threshold.
                  - If so, backtracks to the latest network state and updates the rejection status in the info dictionary.
                  - Records the unsuccessful allocation details for analysis.
                  - Resets the count of unsuccessful allocations to zero for the next SFC.
              """
        if self.count_not_allocated == no_of_steps:  ## if no valid allocation is found after x no_of_steps
            self.vnf_backtrack = deepcopy(self.network)  ## we backtrack `vnf_backtrack` to the latest network state
            info['rejected'] = True  ## update info regarding unsuccessful allocation

            self.recordSfc(sfc, description="not success", place_result=False, route_result=False)
            self.recordInp(info)
            self.nsr_staistics.append(copy.deepcopy(self.record_network))
            ##  initialize the count_not_allocated to zero so that it can be used for another SFC now
            self.count_not_allocated = 0


    def progress_intra_timeslot(self):
        """
            Check if there is a need to invoke the agent in the next step.

            Returns:
                bool: True if the current request is complete, False if not.

            Actions:
                - Checks if the current SFC being processed has reached the last VNF
                - If not, returns False to indicate that the current request is still in progress.
                - If complete, resets the SFC and VNF indices to (0, 0) and returns True
                    to signal the completion of the current request.
            """

        ## The current SFC being processed will have index 'self.sfc_idx' = 0
        ## if 'self.request_batch' has one request its len() will be 1
        if self.sfc_idx < len(self.request_batch):
            ## since self.sfc_idx = 0 and len(self.request_batch) = 1
            ## The current request is not complete, return False
            ## This informs the system that current request is not complete
            return False

        ## The index of self.sfc_idx is 0.
        ## if self.request_batch has one request its len() will be 1
        ## since self.sfc_idx = 0 and (len(self.request_batch)-1) = 0
        ## The current request is complete, return True
        ## This informs the system that current request is complete
        batch_completed = self.sfc_idx >= len(self.request_batch) - 1
        self.sfc_idx, self.vnf_idx = 0, 0  ## SFC and VNF indices are reset to (0, 0).
        return batch_completed  ## Current request is completed return True (here batch_completed value is 1 or True)

    def getNext(self, time_step):
        """
          Get the next set of requests that arrive at the given time step.

          Parameters:
              time_step (int): The current time step in the simulation.

          Returns:
              list: A list containing requests that arrive at the specified time step.

          Actions:
              - Checks if there are one or more requests in the arrival process.
              - Iterates through the requests in the arrival process.
              - For each request, compares its arrival time with the current time step.
              - If the request has arrived at the current time step, adds it to the list and returns the list.
              - If the current time step is less than the arrival time, updates the system time until the arrival time.
              - If multiple requests have the same arrival time, adds all of them to the list.

          Note:
              This function assumes that the arrival process is sorted by arrival time.

          """
        if len(self.arrival_process) >= 1:  ## if self.arrival_process has one or more requests
            wrap_in_list = []
            for index, req in enumerate(self.arrival_process):  ## get the request
                while time_step <= req.arrival_time:  ### if current time_step is less then arrival time
                    if time_step == req.arrival_time:  ## if req1 and req2 arrival is 219 and 220
                        wrap_in_list.append(req)  ## save the request in a list
                        return wrap_in_list
                    else:
                        time_step = self.network.update()  ## update the system time (time_step) to the arrival_time of the request

                        if time_step == req.arrival_time:  ## if time_step is equal to arrival_time
                            wrap_in_list.append(req)  ## save the request in a list
                            return wrap_in_list

    def removeKeyValuePairFromListOfDictionaries(self, list_of_dictionaries, key):
        for d in list_of_dictionaries:
            for i in list(list_of_dictionaries):
                d.pop(key, None)

    def render(self, mode='human'):
        req_batch = str(self.request_batch)

        max_resource = self.vnf_backtrack.calculate_resources()
        avail_resource = self.vnf_backtrack.calculate_resources(False)

        self.removeKeyValuePairFromListOfDictionaries(max_resource, 'icd')
        self.removeKeyValuePairFromListOfDictionaries(avail_resource, 'icd')

        resources = [[num, *[str(avail_res) + ' out of ' + str(max_res) for avail_res, max_res in
                             zip(res[0].values(), res[1].values())]]
                     for num, res in enumerate(
                zip(max_resource, avail_resource))]

        sfcs = [[res.arrival_time, res.ttl, res.bandwidth_demand, res.max_response_latency,
                 '\n'.join([str(vnf).strip("()") for vnf in res.vnfs]), '\n'.join([str(node) for node in nodes])]
                for res, nodes in self.vnf_backtrack.sfc_embedding.items()]
        rep = str(tabulate(resources, headers=[
            'Node', 'Cpu', 'Memory (GiB)', 'Bandwidth (Gbps)', 'ICR (Gbps)', 'Type (Mec/Dc)'], tablefmt="presto"))
        rep += '\n \n \n'

        rep += 'Currently active Service Function Chains in the network:\n'
        rep += str(tabulate(sfcs, headers=['Arrival time (s)', 'TTL (s)', 'Bandwidth (Gbps)',
                                           'Max latency (ms)', 'VNFs (CPU, Memory) -> (num, GiB)', ' Embedding Nodes'],
                            tablefmt="grid"))
        rep += '\n \n'

        rep += 'Last reward = ' + str(self.reward) + '\n'
        rep += 'Current request batch for (network) timestep ' + str(
            self.network.timestep) + ': ' + str(req_batch)

        rep += '\n'

    def compute_state(self, is_reset_called=False, done=False):
        """Compute the environment's state representation."""
        if (done == True):
            return np.asarray([])

        # compute remaining resources of backtrack network i.e. (cpu, memory, bandwidth) for each node
        network_resources = self.vnf_backtrack.calculate_resources(
            remaining=True)

        ### remove 'type' from network_resources. Because it is not included in agent's observation
        self.removeKeyValuePairFromListOfDictionaries(network_resources, 'type')

        network_resources = np.asarray([list(node_res.values())
                                        for node_res in network_resources], dtype=np.float32)


        ## We get max capacity of network resources for each node
        max_resources = self.vnf_backtrack.calculate_resources(remaining=False)

        ### remove 'type' from max_resources. Because it is not included in agent's observation
        self.removeKeyValuePairFromListOfDictionaries(max_resources, 'type')
        # self.removeKeyValuePairFromListOfDictionaries(max_resources, 'icd')

        ## We convert max_resources in to float32 type asarray.
        max_resources = np.asarray([list(node_res.values())
                                    for node_res in max_resources], dtype=np.float32)


        ## We get max value from each colum
        max_resources = np.max(max_resources, axis=0)


        ## We normalize avalible resources (network_resources) by max_resources
        network_resources = network_resources / max_resources


        network_resources = network_resources.reshape(-1)  ## reshaping to 1d vector (flatten)


        if is_reset_called:
            norm_vnf_resources = [0.0, 0.0, 0.0]  ## norm_vnf_resources = (cpu, memory, bandwidth)
            norm_residual_latency = [0.0]
            norm_undeployed = [0.0]
            norm_ttl = [0.0]
            observation = np.concatenate((network_resources,
                                          norm_vnf_resources, norm_residual_latency, norm_undeployed, norm_ttl),
                                         axis=None)
            return observation

        ## compute (normalized) information regarding the VNF which is to be placed next
        sfc = self.request_batch[self.sfc_idx]
        vnf = sfc.vnfs[self.vnf_idx]

        if self.count_not_allocated > 0:

            pass

        norm_vnf_resources = np.asarray([*vnf, sfc.bandwidth_demand])  ## norm_vnf_resources = (cpu, memory, bandwidth)

        norm_vnf_resources = list(
            norm_vnf_resources / max_resources[0:3])  ## max_resources[0:3] = (cpu, memory, bandwidth)
        norm_vnf_resources = list(norm_vnf_resources)

        norm_residual_latency = (sfc.max_response_latency -
                                 self.vnf_backtrack.calculate_current_latency(sfc)) / sfc.max_response_latency


        norm_undeployed = (len(sfc.vnfs) - (self.vnf_idx)) / len(sfc.vnfs)
        norm_ttl = sfc.arrival_time / sfc.ttl
        observation = np.concatenate((network_resources,
                                      norm_vnf_resources, norm_residual_latency, norm_undeployed, norm_ttl), axis=None)

        '''
            We inform the system to invoke the agent:
                for the next VNF of the current SFC or for the next SFC.
                if former:
                    ## select the next VNF of the current SFC:
                if later:
                     ## select the next SFC from the list:
                     ## Chek if the current SFC being processed is completed
                     ## Update the system that "previous_request is completed".
        '''

        if self.is_valid_sfc and not self.is_last_of_sfc:
            ## Tell the system to invoke the agent for the next VNF of the current SFC


            self.sfc_idx, self.vnf_idx = (self.sfc_idx, self.vnf_idx + 1)  ## select the next VNF of the current SFC:

        elif self.is_valid_sfc and self.is_last_of_sfc:
            ## Tell the system to invoke the agent for the next SFC

            self.sfc_idx, self.vnf_idx = (
                self.sfc_idx + 1, 0)  ## select the next SFC from the list: Because we have successful SFC allocation
            ## Chek if the current SFC being processed is  completed, i.e.,
            ## the allocation of all VNFs of current SFC were allocated.
            batch_completed = self.progress_intra_timeslot()

            if batch_completed:
                ## if current_request is completed then current_request becomes previous_request
                ## Update the system that "previous_request is completed". and resulted in "successful SFC allocation"
                self.pre_batch_completed = batch_completed

        elif not self.is_valid_sfc and self.count_not_allocated == 0:
            ## Tell the system to invoke the agent for the next SFC
            self.sfc_idx, self.vnf_idx = (
                self.sfc_idx + 1, 0)  ## select the next SFC from the list: Because we have unsuccessful SFC allocation
            ## Chek if the current SFC being processed is completed:
            ## The function will tell us that SFC being processed is completed because
            ## we manually changed (self.sfc_idx + 1) the index of SFC.
            batch_completed = self.progress_intra_timeslot()

            if batch_completed:
                ## if current_request is completed then current_request becomes previous_request
                ## Update the system that "previous_request is completed" and resulted in un-"successful SFC allocation"
                self.pre_batch_completed = batch_completed

        return observation

    def compute_reward(self, sfc, is_last_of_sfc, is_valid_sfc, batch_completed):
        """Computes the reward dependent on whether a SFC has been succsessfully embedded
        Args:
          sfc: The current SFC
          is_last_of_sfc (bool): determines, if the current sfc is finished i.e. the last vnf was processed
          is_valid_sfc (bool): determines, if the current sfc was valid
          batch_completed (bool): determines, if the current batch of sfc is completed"""

        reward = 0.00


        if is_valid_sfc and is_last_of_sfc:
            reward += 1 / 1000
        elif is_valid_sfc:
            reward += 1 / 2000
        else:
            reward = 0
        return reward

    def printRemainingResources(self):
        resources = [[num, *[str(avail_res) + ' out of ' + str(max_res) for avail_res, max_res in
                             zip(res[0].values(), res[1].values())]]
                     for num, res in enumerate(
                zip(self.vnf_backtrack.calculate_resources(), self.vnf_backtrack.calculate_resources(False)))]
        rep = str(tabulate(resources, headers=[
            'Node', 'Cpu', 'Memory', 'Bandwidth', 'ICD', 'ICR'], tablefmt="presto"))


    ### Experimental:
    def printEachCPUtilization(self):
        """
        this function sums the cpu and sto of VNFs that have been allocated on individual CP
        """
        sum_nodes_cpu = [0.0] * self.num_nodes
        sum_nodes_sto = [0.0] * self.num_nodes

        for res, nodes in self.vnf_backtrack.sfc_embedding.items():
            for index, node in enumerate(nodes):
                tup_val = res.vnfs[index]
                sum_nodes_cpu[node] += tup_val[0]  ## get cpu
                sum_nodes_sto[node] += tup_val[1]  ## get sto

        sum_node_res = list(zip(sum_nodes_cpu, sum_nodes_sto))

    def recordSfc(self, sfc, description, place_result, route_result):
        self.record_network['arrival_time'] = sfc.arrival_time
        self.record_network['ttl'] = sfc.ttl
        self.record_network['ingress'] = sfc.ingress
        self.record_network['egress'] = sfc.egress
        self.record_network['num_vnfs'] = sfc.num_vnfs
        self.record_network['sfc'] = sfc
        self.record_network['bandwidth_demand'] = sfc.bandwidth_demand
        self.record_network['max_response_latency'] = sfc.max_response_latency
        self.record_network['number_of_users'] = sfc.number_of_users
        self.record_network['per_user_datarate'] = sfc.per_user_datarate
        self.record_network['processing_delays'] = sfc.processing_delays
        self.record_network['vnf_datarate'] = sfc.vnf_datarate
        self.record_network['description'] = description
        self.record_network['place_result'] = place_result
        self.record_network['route_result'] = route_result

    def recordInp(self, info):
        try:
            self.record_network['inp_resource_costs'] = (info['resource_costs'])

            self.record_network['inp_cpu_utilization'] = (info['cpu_utilization'])
            self.record_network['inp_memory_utilization'] = (info['memory_utilization'])

            self.record_network['inp_icr_utilization'] = (info['icr_utilization'])
            self.record_network['inp_bandwidth_utilization'] = (info['bandwidth_utilization'])

            self.record_network['inp_operating_servers'] = (info['operating_servers'])
            self.record_network['inp_cp_configurations'] = (info['cp_configurations'])
        except:
            KeyError
