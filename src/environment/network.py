from collections import Counter
from tabulate import tabulate
from pprint import pprint
import networkx as nx
import functools
import operator
import sys
from networkx.exception import NetworkXNoPath
from src.environment.sfc import ServiceFunctionChain


class Network:  ## {'DC': int,'cpu': float, 'memory': GiB,  'icd': ms, 'icr': Gbps, 'price': Euro}
    def __init__(self,
                 overlay,
                 config_price_cp=[{'DC': 0, 'cpu': 36.0, 'memory': 7000.0, 'icd': 1, 'icr': 10.0, 'price': 1.521},
                                  {'DC': 1, 'cpu': 42.0, 'memory': 4000.0, 'icd': 1, 'icr': 10.0, 'price': 1.954},
                                  {'DC': 2, 'cpu': 64.0, 'memory': 14000.0, 'icd': 1, 'icr': 25.0, 'price': 4.173},
                                  {'MEC': 0, 'cpu': 4.0, 'memory': 150.0, 'icd': 1, 'icr': 10.0, 'price': 0.272},
                                  {'MEC': 1, 'cpu': 8.0, 'memory': 300.0, 'icd': 1, 'icr': 25.0, 'price': 0.544},
                                  {'MEC': 2, 'cpu': 16.0, 'memory': 400.0, 'icd': 1, 'icr': 25.0,
                                   'price': 0.768}],
                 config_price_lc=[{'DC': 'DC', 'bandwidth': 10.0, 'icd': 2.8, 'price': 0.247},
                                  {'DC': 'DC', 'bandwidth': 20.0, 'icd': 2.1, 'price': 0.342},
                                  {'DC': 'DC', 'bandwidth': 25.0, 'icd': 4.7, 'price': 0.288},
                                  {'DC': 'MEC', 'bandwidth': 1.0, 'icd': 1.5, 'price': 0.164},
                                  {'DC': 'MEC', 'bandwidth': 4.0, 'icd': 2.7, 'price': 0.247},
                                  {'DC': 'MEC', 'bandwidth': 25.0, 'icd': 3.5, 'price': 0.288},
                                  {'MEC': 'MEC', 'bandwidth': 1.0, 'icd': 1.3, 'price': 0.164},
                                  {'MEC': 'MEC', 'bandwidth': 10.0, 'icd': 1.9, 'price': 0.247},
                                  {'MEC': 'MEC', 'bandwidth': 10.0, 'icd': 3.2, 'price': 0.288},
                                  {'MEC': 'DC', 'bandwidth': 1.0, 'icd': 1.3, 'price': 0.164},
                                  {'MEC': 'DC', 'bandwidth': 10.0, 'icd': 1.9, 'price': 0.247},
                                  {'MEC': 'DC', 'bandwidth': 25.0, 'icd': 3.2, 'price': 0.288}
                                  ]):
        """Internal representation of the network & allocated VNFs.

        Args:
          overlay (str or networkx graph): Path to an overlay graph that specifies the network's properties
                or a valid networkx graph.
          config_price_cp (tuple of float): The cost for one unit of cpu, memory and bandwidth"""

        # parse and validate the overlay network
        self.overlay, properties = Network.check_overlay(overlay)
        self.num_nodes = properties['num_nodes']
        self.timestep = 0
        self.cp_configurations = config_price_cp
        self.lc_configurations = config_price_lc

        self.list_of_mec_cp = properties['list_of_mec_cp']
        self.list_of_dc_cp = properties['list_of_dc_cp']
        self.list_of_all_cp = properties['list_of_all_cp']

        # the service function chains with its mapping to the network nodes:
        self.sfc_embedding = dict()

        self.save_cp_configurations = {key: None for key in self.list_of_all_cp}
        self.save_lc_configurations = int

    @staticmethod
    def check_overlay(overlay):
        """
           Checks whether the overlay adheres to the expected parameter types and attributes and returns the parsed
           network instance along with computed properties.

           Parameters:
           - overlay: str or networkx.Graph
               If a string ending with '.gpickle', the function reads the Graph from the file.
               Otherwise, it expects a networkx.Graph instance.

           Returns:
           tuple (networkx.Graph, dict)
               Returns a tuple containing the parsed network instance and a dictionary of computed properties.

           Raises:
           AssertionError:
               Raised if the overlay does not adhere to the expected attributes and data types.
           """

        # parse overlay from gpickle if
        if isinstance(overlay, str) and overlay.endswith('.gpickle'):
            overlay = nx.read_gpickle(overlay)

        node_attributes = {'cpu': float, 'memory': float, 'bandwidth': float, 'icd': float, 'icr': float}

        for _, data in overlay.nodes(data=True):
            assert (all([nattr in data for nattr, ntype in node_attributes.items(
            )])), 'Overlay must specify all required node attributes.'
            assert (all([type(data[nattr]) == ntype for nattr, ntype in node_attributes.items(
            )])), 'Overlay must specify the correct data types.'

        edge_attributes = {'latency': float}
        for _, _, data in overlay.edges(data=True):
            assert (all([eattr in data for eattr, etype in edge_attributes.items(
            )])), 'Overlay must specify all required edge attributes.'
            assert (all([type(data[eattr]) == etype for eattr, etype in edge_attributes.items(
            )])), 'Overlay must specify the correct data types.'

        # compute properties of the parsed graph
        properties = {}
        properties['num_nodes'] = overlay.number_of_nodes()
        _, resource = next(iter(overlay.nodes(data=True)))
        properties['num_node_resources'] = len(resource)
        list_of_mec_cp = []
        list_of_dc_cp = []

        for node_num, node_properties in overlay.nodes(data=True):
            if node_properties['type'] == 'MEC':
                list_of_mec_cp.append(node_num)
            elif node_properties['type'] == 'DC':
                list_of_dc_cp.append(node_num)

        # populate properties dictionary with computed values
        properties['list_of_mec_cp'] = list_of_mec_cp
        properties['list_of_dc_cp'] = list_of_dc_cp
        properties['list_of_all_cp'] = list_of_mec_cp + list_of_dc_cp
        properties['num_of_mec_cp'] = len(list_of_mec_cp)
        properties['num_of_dc_cp'] = len(list_of_dc_cp)

        return overlay, properties

    def update(self, time_increment=1):
        """Let the network run for the given time"""
        self.timestep += time_increment

        # delete all SFCs that exceed their TTL
        def check_ttl(sfc):
            return sfc.ttl >= self.timestep

        # filter SFCs based on TTL and update the sfc_embedding
        sfc_embedding = {sfc: nodes for sfc, nodes in self.sfc_embedding.items() if check_ttl(sfc)}
        self.sfc_embedding = sfc_embedding

        return self.timestep

    def embed_vnf(self, sfc: ServiceFunctionChain, vnf: tuple, node: int):

        """
            Embeds the Virtual Network Function to a specified computing platform.

            Parameters:
            - sfc: ServiceFunctionChain
                The Service Function Chain to which the VNF belongs.
            - vnf: tuple
                Tuple representing the Virtual Network Function (VNF) to be embedded.
            - node: int
                The computing platform (node) to which the VNF should be embedded.

            Returns:
            bool
                True if the VNF is successfully embedded, False otherwise.
        """

        # Reject embedding if the specified node is out of bounds
        if node >= self.num_nodes:
            return False

        # Reject embedding if the network does not provide sufficient VNF resources
        # or if the action voluntarily chooses not to embed
        if not self.check_vnf_resources(vnf, sfc, node):
            return False

        # If the SFC is not already in the embedding dictionary, add it
        if sfc not in self.sfc_embedding:
            self.sfc_embedding[sfc] = []

        # Append the node to the list of embedded nodes for the specified SFC
        self.sfc_embedding[sfc].append(node)
        return True

    def updateCpuStoAndBandwidth(self, resources, nodes, sfc):

        for vnf_idx, node_idx in enumerate(nodes):
            # Update CPU and memory resources based on VNF requirements
            resources[node_idx]['cpu'] -= sfc.vnfs[vnf_idx][0]
            resources[node_idx]['memory'] -= sfc.vnfs[vnf_idx][1]

            ### Bandwidth is demanded if successive VNFs are allocated on different computing platforms.
            ### Except for the last VNF of the SFC chain, for whom we always demand bandwidth
            if vnf_idx == len(nodes) - 1:
                resources[node_idx]['bandwidth'] -= sfc.bandwidth_demand
            elif not nodes[vnf_idx] == nodes[vnf_idx + 1]:
                resources[node_idx]['bandwidth'] -= sfc.bandwidth_demand

    def updateResourceUsage(self, inp_resources):
        """
        Args:
        inp_resources(list[dict{}]):
        """
        for sfc, nodes in self.sfc_embedding.items():

            # Update CPU, memory, bandwidth, and icr based on the VNF allocation for each SFC
            self.updateCpuStoAndBandwidth(inp_resources, nodes, sfc)
            ### icr is demanded if the VNF is placed on the same server as the previous VNF
            for vnf_idx, inp_cp_number in enumerate(nodes):
                if vnf_idx > 0:
                    if nodes[vnf_idx] == nodes[vnf_idx - 1]:  ## if curr_cp == prev_cp
                        inp_resources[inp_cp_number]['icr'] -= sfc.bandwidth_demand

    def getResources2(self, overlay):
        """
        ## This function converts networx topology into list of dictionaries:

        networkx =     [(0, {'cpu': 64.0, 'memory': 14000.0, 'bandwidth': 25.0, 'icd': 10.0, 'icr': 25.0}),
                        (1, {'cpu': 64.0, 'memory': 14000.0, 'bandwidth': 25.0, 'icd': 20.0, 'icr': 25.0})]
        list_of_dict = [{'cpu': 64.0, 'memory': 14000.0, 'bandwidth': 25.0, 'icd': 10.0, 'icr': 25.0},
                        {'cpu': 64.0, 'memory': 14000.0, 'bandwidth': 25.0, 'icd': 20.0, 'icr': 25.0},]

        ## To access node properties in a list_of_dict.
        list_of_dict[0]['CPU'] = 64.0 ## refers to the node(Computing Platform) zero(0) 'CPU' value.

        ## The conversion from networx topology into list of dictionaries is modified.
        The cp number is added to its corresponding index in the list_of_dict.
        Example:
            list of All CP:  [3, 0, 2, 1]
            list of MEC CP:  [3]
            If CP MEC number is 3 then it should be added to index 3 of the list_of_dict
        Returns:
        None
        """
        inp = [None] * (overlay.number_of_nodes())  ## create a list of CPs

        for node_num, node_properties in self.overlay.nodes(data=True):

            temp_dict = dict()  ## add the node properties to the dictionary
            for node_properties, val in node_properties.items():
                temp_dict[node_properties] = val
            inp[node_num] = temp_dict

        return inp

    def calculate_resources(self, remaining=True) -> list:
        """Calculates the remaining resources for all computing platforms.

        Returns:
            List of dictionaries representing the resources of each computing platform.
        """

        inp_resources = self.getResources2(self.overlay)
        if remaining:
            # calculated remaining resources
            self.updateResourceUsage(inp_resources)
        return inp_resources

    def checkCurrentAndPreviousVnfCp(self, curr_cp) -> bool:
        """
            Checks if the current computing platform (cp) is the same as the previous one

            Returns:
            True if the current cp is the same as the previous cp, False otherwise.
            """
        for sfc, nodes in self.sfc_embedding.items():
            ## the current_cp is not yet stored in sfc_embedding

            def index_in_list(a_list, index) -> bool:
                if len(a_list) > 0: return index < len(a_list)  ## check if one element exists in a list

            if index_in_list(nodes, -1):
                prev_cp = nodes[-1]  ## get current prev_cp
                if prev_cp == curr_cp:  ## if curr_cp == prev_cp
                    return True if prev_cp == curr_cp else False

    def check_sfc_constraints(self, sfc, cp_sampled):
        """ Check whether the (partial) SFC embedding satisfies the bandwidth and latency constraints, i.e. if the
        bandwidth demand can be satisfied by a CP (cp_sampled) and if the SFC can still satisfy its latency
        constraints.
        """

        ## check if current cp (cp_sampled) can cover the bandwidth demand of the request
        inp_resources_available = self.calculate_resources()

        bandwidth_constraint = sfc.bandwidth_demand <= inp_resources_available[cp_sampled]['bandwidth']

        if not bandwidth_constraint:
            # Remove the CP number if bandwidth constraint is not satisfied
            ## When constraints such as CPU, STO and ICR are satisfied for a
            #  particular CP the CP number is saved in sfc_embedding[sfc]
            ## Since bandwidth_constraint is NOT satisfied therefore we need
            #  to remove the CP number that was saved earlier.
            self.sfc_embedding[sfc].pop(-1)
            return False

        ## Verify the latency constraints for a (partially) allocated SFC, i.e.
        ## does the latency of prior VNF allocations violate the maximum latency of the NSR?
        try:
            latency = self.calculate_current_latency(sfc)
            latency_constraint = latency <= sfc.max_response_latency

            if not latency_constraint:
                # When constraints such as latency is satisfied for a particular CP the CP number is saved in
                # sfc_embedding[sfc]. Since residual_latency is NOT satisfied therefore we need to remove the CP
                # number that was saved earlier.
                self.sfc_embedding[sfc].pop(-1)
                return False

        except NetworkXNoPath:
            # Set latency_constraint to False if no path is found
            latency_constraint = False

        # The CP number is saved in sfc_embedding[sfc] if constraints such as (CPU, STO and ICR) are satisfied for
        # the current VNF # Next we check constraints such as (bandwidth_constraint and latency_constraint) if one of
        # them is NOT satisfied therefore we need to remove the CP number that was saved earlier. # However,
        # the LC price configuration and data-rate is different between MEC-MEC and MEC-DC. # Therefore, we need to
        # confirm here if the LC price configuration for MEC-MEC and MEC-DC is available.
        if sfc in self.sfc_embedding:
            sfc_vnf_cp_list = self.sfc_embedding[sfc]  ## list of CPs in which the VNFs of an SFC are allocated.

            if len(sfc_vnf_cp_list) > 1:  ## If sfc_vnf_cp_list has one CP do nothing. A LC is formed between two CPs.
                inp_resources_utilization = self.calculate_resource_utilization()
                cp_pair_num = self.getSfcLcNum(sfc_vnf_cp_list)
                cp_pair_typ = self.getSfcLcTyp(cp_pair_num)

                for cp_pair_nu, cp_pair_ty in zip(cp_pair_num, cp_pair_typ):
                    cp_one = cp_pair_nu[0]
                    cp_one_type = cp_pair_ty[0]
                    cp_two_type = cp_pair_ty[1]
                    lc_config_price = self.whichLcConfiguration(inp_resources_utilization[cp_one], cp_one_type,
                                                                cp_two_type, print_lc_config=False)
                    if lc_config_price is None:
                        # Since LC price configuration between two CP types is NOT available therefore we need to
                        # remove the CP number that was saved earlier.
                        self.sfc_embedding[sfc].pop(-1)
                        return False
        return bandwidth_constraint and latency_constraint

    def getIcd(self, sfc, nodes):
        resources = self.getResources2(self.overlay)
        for node_idx, node_num in enumerate(nodes):  ## list of nodes in which the VNFS of an SFC are allocated
            if node_idx > 0:
                if nodes[node_idx] == nodes[node_idx - 1]:  ## if curr_cp == prev_cp
                    icd = resources[node_num]['icd']
                    sfc.processing_delays[node_idx] = icd

    def calculate_current_latency(self, sfc):
        """Calculates the current latency of the SFC i.e the end-to-end delay from the start of the SFC to the currently
            last allocated VNF of the SFC.
            Throws NetworkXNoPath, if there is no possible path between two VNFs
            If current VNF is allocated in a node where previous VNF was allocated then use current nodes icd
            """
        latency = 0

        ## compute transmission delay if the SFC is already (partially) allocated
        if sfc in self.sfc_embedding:
            nodes = self.sfc_embedding[sfc]  ## list of nodes in which the VNFS of an SFC are allocated
            ## First calculate the latency over lc if two VNFs are allocated on different nodes
            latency = sum([nx.dijkstra_path_length(self.overlay, nodes[idx - 1],
                                                   nodes[idx], weight='latency') for idx in range(1, len(nodes))])

            ## compute the icd delay if the SFC is already (partially) allocated
            self.getIcd(sfc, nodes)
            latency += sum(sfc.processing_delays[:len(nodes)])

        return latency

    def check_embeddable(self, sfc: ServiceFunctionChain, cp_sampled):
        """ Check whether the (partial) SFC embedding can still satisfy its constraints
        """

        ## check whether SFC can still fulfill service constraints (SFC constraints)
        sfc_constraints = self.check_sfc_constraints(sfc, cp_sampled)
        return sfc_constraints

    def constraintsCpuAndStorage(self, cp_cpu, cp_sto, vnf_cpu, vnf_sto):

        if cp_cpu >= vnf_cpu:
            if cp_sto >= vnf_sto:
                return True
            else:
                return False
        else:
            return False

    def checkConstraintIcr(self, cp_icr, bw):
        if cp_icr >= bw:
            return True
        else:
            return False

    def check_vnf_resources(self, vnf, sfc, inp_cp_num=None):
        """
        Check if the specified CP (inp_cp_num) provides enough resources to allocate current VNF

        :param vnf (tuple) vnf[0] has the value of CPU required by vnf
                           vnf[1] has the value of STO required by vnf
        :param sfc (object)
        :param inp_cp_num (int) Number of CP in which the VNF should be allocated
        """

        inp_resources = self.calculate_resources()

        if inp_cp_num is not None:
            cp_resources = inp_resources[inp_cp_num]

        cpu_sto_constraint = self.constraintsCpuAndStorage(cp_cpu=cp_resources['cpu'], cp_sto=cp_resources['memory'],
                                                           vnf_cpu=vnf[0], vnf_sto=vnf[1])

        is_empty = not bool(self.sfc_embedding)  # Check if self.sfc_embedding (dict) is empty

        ## if self.sfc_embedding is empty (True) then do not check icr. Because this is first VNF of an SFC.

        """
         :param check_icr_constraint (bool) True if the current VNF is being allocated in the same CP as the previous VNF.
         :param icr_constraint: (bool) is True if a CP's icr can meet the bandwidth requirements of a VNF data rate.
        """
        check_icr_constraint = None
        icr_constraint = None

        if not is_empty:  ## If sfc_embedding is not empty then check icr. Because previous VNF allocation in CP exists
            vnf_pre_and_cur = self.checkCurrentAndPreviousVnfCp(inp_cp_num)
            if vnf_pre_and_cur:  ## If current VNF is being allocated in a CP where Previous VNF was allocated
                icr_constraint = self.checkConstraintIcr(cp_resources['icr'], sfc.bandwidth_demand)
                check_icr_constraint = True
        if check_icr_constraint:
            return cpu_sto_constraint and icr_constraint  # return True if both boolean variables are true; else, false.
        elif check_icr_constraint is None:  ## if icr_constraint is None return only the {cpu and sto}
            return cpu_sto_constraint  ## True/False

    def calculate_occupied_resources(self):
        """Calculates a dictionary that summarizes the amount of occupied resources per type."""

        ## compute the amount of maximum resources / available resources per node
        resources = self.calculate_resources(remaining=False)  ## get maximum resources
        avail_resources = self.calculate_resources(remaining=True)  ## get remaining resources

        pprint(resources)
        pprint(avail_resources)

        # reduce the respective resources over the entire network (i.e. summed over all nodes)
        resources = dict(functools.reduce(operator.add,
                                          map(Counter, resources)))
        avail_resources = dict(functools.reduce(operator.add,
                                                map(Counter, avail_resources)))

        # get the amount of depleted resources and their costs
        depleted = {key: resources[key] -
                         avail_resources[key] for key in resources}
        costs = {key: depleted[key]
                      * self.cp_configurations[key] for key in depleted}

        return costs

    def subtractValuesFromListOfDictionaries(self, base, subtract):
        """
        Subtract values from list of dictionaries from another list of dictionaries
        https://stackoverflow.com/questions/33284460/subtract-values-from-list-of-dictionaries-from-another-list-of-dictionaries
        Args:
            base:
            subtract:
        """
        corrected = []
        for base_dict, sub_dict in zip(base, subtract):
            ## If rounding the values to two decimal places use
            # corrected.append({key: round(val - sub_dict.get(key, 0), 2) for key, val in base_dict.items()})
            corrected.append({key: val - sub_dict.get(key, 0) for key, val in
                              base_dict.items()})  ## without rounding the values to any decimal places
        return corrected

    def removeKeyValuePairFromListOfDictionaries(self, list_of_dictionaries, key):
        for d in list_of_dictionaries:
            for i in list(list_of_dictionaries):
                d.pop(key, None)

    def calculate_resource_utilization(self):
        """Calculates a dictionary that summarizes the resource utilization per resource type.
        The resource type we consider are: 'cpu', 'memory', 'bandwidth', and 'icr'
        The 'type' and 'icd' are not relevant resources for this function.
        Therefore, we remove the 'icd' and 'type'
        Return: A list of dictionary
        """

        ### compute the amount of maximum resources / available resources per node
        max_resources = self.calculate_resources(remaining=False)  ## amount of maximum resources per node
        avail_resources = self.calculate_resources(remaining=True)  ## amount of available resources per node

        ### remove 'icd' from resources
        self.removeKeyValuePairFromListOfDictionaries(max_resources, 'icd')
        self.removeKeyValuePairFromListOfDictionaries(avail_resources, 'icd')
        ### remove 'type' from resources
        self.removeKeyValuePairFromListOfDictionaries(max_resources, 'type')
        self.removeKeyValuePairFromListOfDictionaries(avail_resources, 'type')

        ##  amount of utilized resources per node
        utilization = self.subtractValuesFromListOfDictionaries(max_resources,
                                                                avail_resources)

        return utilization

    def getInpUtilization(self, avail_resources, max_resources):
        resources = [
            [num, *[str(avail_resources) + ' out of ' + str(max_resources) for avail_resources, max_resources in
                    zip(res[0].values(), res[1].values())]]
            for num, res in enumerate(zip(avail_resources, max_resources))]
        rep = str(tabulate(resources, headers=[
            'Node', 'Cpu', 'Memory', 'Bandwidth', 'ICD', 'ICR'], tablefmt="presto"))
        print(rep)

    def get_operating_nodes(self):
        """Computes the set of indices that describe all operating nodes."""
        operating_nodes = {server for sfc in self.sfc_embedding for server in self.sfc_embedding[sfc]}
        return operating_nodes


    def whichCpConfiguration(self, cp_num, cp, cp_type):
        for cp_config in self.cp_configurations:
            cp_config_type = next(
                iter(cp_config))  ## Getting first key in dictionary, in this case it will be MEC or DC
            if cp_type == cp_config_type:
                if cp['cpu'] <= cp_config['cpu'] and cp['memory'] <= cp_config['memory'] and cp['icr'] <= cp_config[
                    'icr']:
                    ## We save only cp_config number
                    self.save_cp_configurations[cp_num] = (int(cp_config[cp_type]))
                    return cp_config['price']

    def whichLcConfiguration(self, cp, cp_one_type, cp_two_type, print_lc_config=False):

        lc_config_number = -1
        for lc_config in self.lc_configurations:
            first_key_value_pair = next(iter(lc_config.items()))  # Get first key-value pair of the dictionary

            if cp_one_type == first_key_value_pair[0] and cp_two_type == first_key_value_pair[1]:
                lc_config_number += 1
                if cp['bandwidth'] <= lc_config['bandwidth']:
                    if print_lc_config:
                        self.save_lc_configurations = lc_config_number
                    return lc_config['price']

    def getSfcLcNum(self, cp_list):
        """
        This function takes in a list of CPs. Converts the list into a list of tuples
        cp_list [] (int): list of CPs
        cp_pair [] (int): returns a pair of CPs
        Example:
           Input:  cp_list:  [1, 0, 3, 1, 2]
           Output: cp_pair:  [(1, 0), (0, 3), (3, 1), (1, 2)]
        """
        cp_pair = []
        for node_idx, node_num in enumerate(cp_list):  ## list of nodes in which the VNFS of an SFC are allocated
            if node_idx >= 1:  ## a link is formed between two nodes. If list has one node do nothing.
                cur_cp = cp_list[node_idx]
                pre_cp = cp_list[node_idx - 1]
                cp_pair.append((pre_cp, cur_cp))
        return cp_pair

    def getSfcLcTyp(self, cp_pair_num):
        """
        This function takes in a list of tuples. Converts the list into a list of tuples
        cp_pair_num [] (int): list containing tuples of CPs
        cp_pair_type [] (str): returns a list of CP type (MEC or DC) tuples
        Example:
          Input:  cp_list:  [1, 0, 3, 1, 2]
          Output: cp_pair_type:  [('DC', 'DC'), ('DC', 'MEC'), ('MEC', 'DC'), ('DC', 'DC')]
          Assumption: 0, 1 and 2 are DC and 3 is MEC
               """

        cp_pair_typ = []
        for index, _tuple in enumerate(cp_pair_num):
            cp_one = _tuple[0]
            cp_two = _tuple[1]

            cp_one_type = self.currentCpNumberMecOrDc(cp_one)
            cp_two_type = self.currentCpNumberMecOrDc(cp_two)

            cp_pair_typ.append((cp_one_type, cp_two_type))
        return cp_pair_typ

    def calculate_resource_costs(self, sfc: ServiceFunctionChain):
        """Computes a dictionary that summarizes the current operation costs per resource type."""

        record_network = {}
        ## get a set of currently operating computing platforms
        operating_cp = self.get_operating_nodes()

        record_network['operating_nodes'] = list(operating_cp)

        if not operating_cp:
            self.cp_configurations = dict.fromkeys(self.list_of_all_cp, None)

            return {'resource': 0.0}, self.cp_configurations, record_network

        resource_utilization = self.calculate_resource_utilization()

        cp_config_cost = 0.0

        for cp_num, cp in enumerate(resource_utilization):
            if not all(value == 0.0 for value in cp.values()):
                cp_type = self.currentCpNumberMecOrDc(cp_num)
                cp_config_price = self.whichCpConfiguration(cp_num, cp, cp_type)

                try:
                    cp_config_cost += cp_config_price
                except:
                    print("Unable to find CP configuration with price\n"
                          "Check the actual configuration of CP and its configuration price")
                    sys.exit("Error message")

        cp_config_cost = round(cp_config_cost, 2)

        record_network['cp_configurations'] = self.save_cp_configurations
        record_network['cp_config_cost'] = round(cp_config_cost, 2)

        ## compute lc cost if the SFC is already (partially) allocated
        lc_config_cost = 0.0
        lc_pair_config = dict()
        if sfc in self.sfc_embedding:
            cp_pair_num = self.getSfcLcNum(self.sfc_embedding[sfc])
            cp_pair_typ = self.getSfcLcTyp(cp_pair_num)
            record_network['nsr_vl_num'] = cp_pair_num
            record_network['nsr_vl_cp_type'] = cp_pair_typ
            for cp_pair_nu, cp_pair_ty in zip(cp_pair_num, cp_pair_typ):
                cp_one = cp_pair_nu[0]
                cp_two = cp_pair_nu[1]
                cp_one_type = cp_pair_ty[0]
                cp_two_type = cp_pair_ty[1]
                if cp_one != cp_two:  ## If cp one == cp two,
                    ## the previous and current VNF are allocated in same cp,
                    ## so the lc is not required and thus lc cost=0.0.
                    for cp_num, cp in enumerate(resource_utilization):
                        if cp_one == cp_num:
                            if not all(value == 0.0 for value in cp.values()):
                                lc_config_price = self.whichLcConfiguration(cp, cp_one_type, cp_two_type,
                                                                            print_lc_config=True)
                                lc_pair_config[(cp_one, cp_two)] = self.save_lc_configurations
                                try:
                                    lc_config_cost += lc_config_price
                                except:
                                    print("Unable to find LC configuration with price\n"
                                          "Check the actual configuration of LC and its configuration price")
                                    sys.exit("Error message")

        lc_config_cost = round(lc_config_cost, 2)
        total_cost = cp_config_cost + lc_config_cost

        record_network['lc_config'] = lc_pair_config
        record_network['lc_config_cost'] = lc_config_cost

        return {'resource': total_cost}, self.save_cp_configurations, record_network  ## default

    def currentCpNumberMecOrDc(self, cp_num):
        if cp_num in self.list_of_mec_cp:
            return 'MEC'
        elif cp_num in self.list_of_dc_cp:
            return 'DC'
