import os
import sys
import csv
import json
import heapq
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from abc import abstractmethod
from collections.abc import Generator
from src.environment.sfc import ServiceFunctionChain


class ArrivalProcess(Generator):
    """Abstract class that implements the request generation. Inheriting classes implement the parameterization of to
    be generated SFCs. """

    def __init__(self, **kwargs):
        self.timeslot = 1

        # Generate requests according to the abstract method `generate_requests`.
        self.requests = self.generate_requests()

        # Order arriving SFCs according to their arrival time.
        self.requests = [((sfc.arrival_time, num), sfc) for (num, sfc) in enumerate(self.requests)]
        heapq.heapify(self.requests)

    def send(self, arg):
        """Implements the generator interface. Generates a list of arriving SFC requests for the
        respective timeslot starting from timeslot one.
        """
        req = []

        if len(self.requests) <= 0:
            self.throw()

        # Add SFCs to the batch until the next SFC has an arrival time that exceeds the internal timestep.
        while len(self.requests) > 0 and self.requests[0][1].arrival_time <= self.timeslot:
            _, sfc = heapq.heappop(self.requests)
            req.append(sfc)

        # Increment internal timeslot.
        self.timeslot += 1
        return req

    def throw(self, type=None, value=None, traceback=None):
        """Raises an Error if all SFCs were already generated."""
        raise StopIteration

    @staticmethod
    def factory(config):
        """
            Factory method to allow for an easier generation of different arrival processes.

            Parameters:
            - config (dict): Configuration dictionary containing information about the arrival process.

            Returns:
            - ArrivalProcess: Instance of the appropriate arrival process class.
        """

        if 'type' not in config:
            arrival = JSONArrivalProcess(config)

        arrival_type = config['type']
        params = {key: value for key, value in config.items() if key != 'type'}

        # Set a seed solely if the episodes should be static,
        # i.e., should always get the same requests for each episode.

        if params['static_arrival_time_per_episode']:
            params['seed'] = False
            # print(params['static_arrival_time_per_episode'])
        else:
            params['seed'] = False

        if arrival_type == 'poisson_arrival':
            arrival = PoissonArrivalUniformLoad(**params)

        else:
            raise ValueError('Unknown arrival process')

        return arrival

    @abstractmethod
    def generate_requests(self):
        """Abstract method that must generate a list of SFCs to be emitted."""
        raise NotImplementedError('Must be overwritten by an inheriting class')


class JSONArrivalProcess(ArrivalProcess):
    def __init__(self, request_path):
        """Instantiate an arrival process that generates SFC requests at their respective arrival timeslot from a
        specified JSON file.

        Parameters:
        - request_path (str): Path to the JSON file containing SFC parameters.
        """
        assert (isinstance(request_path, str))
        assert (request_path.endswith('.json'))
        self.request_path = request_path
        super().__init__()

    def generate_requests(self):
        """Generates SFC objects according to their paramterization as given by a JSON file."""
        # load SFCs from specified JSON file
        with open(self.request_path, 'rb') as read_file:
            requests = json.load(read_file)

        def parse_vnfs(vnfs):
            return [tuple(vnf.values()) for vnf in vnfs]

        # create SFC objects with parameters specified in the JSON file
        req = []
        for sfc in requests:
            vnfs = parse_vnfs(sfc.pop('vnfs'))
            sfc = ServiceFunctionChain(vnfs=vnfs, **sfc)
            req.append(sfc)
        return req


class StochasticProcess(ArrivalProcess):
    """Generates SFC requests using a stochastic process based on a load generator."""

    def __init__(self, num_requests, load_generator):
        """Instantiate a stochastic process for generating SFC requests.

              Parameters:
              - num_requests (int): Number of SFC requests to generate.
              - load_generator (LoadGenerator): Load generator for generating SFC parameters.
              """

        self.num_requests = num_requests
        self.load_generator = load_generator
        super().__init__()

    def generate_requests(self):
        """Generates SFC requests based on a stochastic process."""
        load_gen = self.load_generator.next_sfc_load()
        arrival_gen = self.next_arrival()

        req = []
        while len(req) < self.num_requests:
            arrival_time, ttl = next(arrival_gen)
            sfc_params = next(load_gen)
            sfc = ServiceFunctionChain(arrival_time=arrival_time, ttl=ttl, **sfc_params)
            req.append(sfc)

        # print(req)
        return req


def calculateRequestArrivalRatePerTimeSlot(timeslot_length, request_arrival_per_timeslot):
    request_arrival_rate_per_timeslot = request_arrival_per_timeslot / timeslot_length
    return float(request_arrival_rate_per_timeslot)


def calculateNumberOfRequestsPerTimeSlot(timeslot_length, request_arrival_rate_per_timeslot):
    request_arrival_per_timeslot = request_arrival_rate_per_timeslot * timeslot_length
    return int(request_arrival_per_timeslot)


def calculateRequestDeletionRatePerTimeSlot(timeslot_length, request_deletion_in_timeslots):
    request_deletion_rate = 1 / (request_deletion_in_timeslots * timeslot_length)
    return float(request_deletion_rate)


def requestArrival(service):
    timeslot_length = service['timeslot_length']

    if service['request_arrival_rate'] == -1 and service['request_arrival_per_timeslot'] == -1:
        sys.exit("Error: Please specify: request_arrival_rate OR request_arrival_per_timeslot ")

    if service['request_arrival_rate'] > 0 and service['request_arrival_per_timeslot'] > 0:
        sys.exit("Error: Both Specified: request_arrival_rate AND request_arrival_per_timeslot ")

    if service['request_arrival_rate'] == -1 and service['request_arrival_per_timeslot'] > 0:
        request_arrival_rate = calculateRequestArrivalRatePerTimeSlot(timeslot_length,
                                                                      service['request_arrival_per_timeslot'])
        return request_arrival_rate, service['request_arrival_per_timeslot']

    if service['request_arrival_per_timeslot'] == -1 and service['request_arrival_rate'] > 0:
        request_arrival_per_timeslot = calculateNumberOfRequestsPerTimeSlot(timeslot_length,
                                                                            service['request_arrival_rate'])
        return service['request_arrival_rate'], request_arrival_per_timeslot


def requestDeletion(service):
    timeslot_length = service['timeslot_length']

    if service['request_deletion_rate'] == -1 and service['request_deletion_in_timeslots'] == -1:
        sys.exit("Error: Please specify: request_deletion_rate OR request_deletion_in_timeslots ")

    if service['request_deletion_rate'] > 0 and service['request_deletion_in_timeslots'] > 0:
        sys.exit("Error: Both Specified: request_deletion_rate AND request_deletion_in_timeslots ")

    if service['request_deletion_rate'] == -1 and service['request_deletion_in_timeslots'] != -1:
        request_deletion_rate_beta = calculateRequestDeletionRatePerTimeSlot(timeslot_length,
                                                                             service['request_deletion_in_timeslots'])
        return request_deletion_rate_beta, service['request_deletion_in_timeslots']

    if service['request_deletion_in_timeslots'] == -1 and service['request_deletion_rate'] != -1:
        print("Enter request_deletion_time_in_timeslots")


class PoissonArrivalUniformLoad(StochasticProcess):
    def __init__(self, ingress, egress, req_num_of_vnfs, req_vnf_names,
                 initial_number_of_users, min_and_max_number_of_users, per_user_data_rate,
                 max_response_latency, seed,
                 **service):

        seed = False
        if seed:
            random.seed(True)
        else:
            random.seed(seed)

        ### pre-processing
        request_arrival_rate, request_arrival_per_timeslot = requestArrival(service)
        request_deletion_rate, request_deletion_timeslot = requestDeletion(service)
        timeslot_length = service['timeslot_length']
        num_of_timeslots = service['num_of_timeslots']

        '''
        print("timeslot_length:", timeslot_length)
        print("num_of_timeslots:", num_of_timeslots)
        print("request_arrival_rate", request_arrival_rate, "\nrequest_arrival_per_timeslot",
              request_arrival_per_timeslot)
        print("request_deletion_rate", request_deletion_rate, "\nrequest_deletion_timeslot", request_deletion_timeslot)
        print("total_num_requests:", request_arrival_per_timeslot * num_of_timeslots)
        # input()

        '''
        total_num_requests = request_arrival_per_timeslot * num_of_timeslots
        self.num_requests = total_num_requests
        self.num_timeslots = num_of_timeslots * timeslot_length

        # derive parametrisation of arrival- & service-time distribution

        self.mean_arrival_rate = request_arrival_rate
        self.mean_service_rate = []
        self.mean_service_rate.append(request_deletion_timeslot * timeslot_length)
        self.mean_service_rate.append(request_deletion_timeslot)
        self.mean_service_rate.append(timeslot_length)

        # generate SFC & VNF parameters uniformly at random within their bounds
        load_generator = UniformLoadGenerator(ingress, egress, req_num_of_vnfs, req_vnf_names,
                                              initial_number_of_users, min_and_max_number_of_users, per_user_data_rate,
                                              max_response_latency)

        super().__init__(self.num_requests, load_generator)

    def next_arrival(self):

        # interarrival times conform to an exponential distribution with the rate parameter `mean_service_rate`
        # nsr_arrival_time = [(numpy.random.exponential(self.mean_arrival_rate)) for _ in range(self.num_requests)]
        nsr_arrival_time = [random.expovariate(self.mean_arrival_rate) for _ in range(self.num_requests)]
        nsr_arrival_time = np.ceil(np.cumsum(nsr_arrival_time))
        nsr_arrival_time = nsr_arrival_time.astype(int)

        # service times conform to an exponential distribution with the rate parameter `1 / mean_service_rate`

        request_deletion_timeslot = self.mean_service_rate[1]
        timeslot_length = self.mean_service_rate[2]
        nsr_deletion_time = []
        for val in (nsr_arrival_time):
            nsr_deletion_time.append(val + (request_deletion_timeslot * timeslot_length))

        for arrival_time, service_time in zip(nsr_arrival_time, nsr_deletion_time):
            yield arrival_time, service_time


class UniformLoadGenerator:
    def __init__(self, ingress, egress, req_num_of_vnfs, req_vnf_names,
                 initial_number_of_users, min_and_max_number_of_users, per_user_data_rate,
                 max_response_latency):
        # SFC object generation parameters

        # self.request_name = request_name
        self.req_num_of_vnfs = req_num_of_vnfs
        self.req_vnf_names = req_vnf_names
        self.initial_number_of_users = initial_number_of_users
        self.min_and_max_number_of_users = min_and_max_number_of_users
        self.max_response_latency = max_response_latency
        self.per_user_data_rate = per_user_data_rate
        self.ingress = ingress
        self.egress = egress

    def GetSliceTypePerUserDatarate(self, per_user_data_rate):  # in Mbps:

        # return int(numpy.random.uniform(*self.per_user_data_rate))
        return int(random.uniform(*self.per_user_data_rate))

    def GetSliceTypeAndNumberOfUsers(self, initial_number_of_users, min_and_max_number_of_users):

        if self.initial_number_of_users == -1 and min_and_max_number_of_users:
            return int(random.uniform(*self.min_and_max_number_of_users))
        else:
            if self.initial_number_of_users != -1:
                return self.initial_number_of_users

    def CalculateStorageOfVnf(self, req_bandwidth_of_vnf, vnf_storage):
        """
        Args:
              vnf_storage (float): Storage required by a specific VNF in GB
              req_bandwidth_of_vnf (float): in Mbps
        return:
              vnf_storage (float) in GiB
              """
        ## vnf_storage is in GB and the computing platform storage is provided in GiB
        ## convert GB to GiB  ------  1.0 GB  == 0.931323 GiB
        ## divide the value of vnf_storage in GB by 1.074 to get the value of vnf_storage in GiB
        vnf_storage = (vnf_storage / 1.074)
        # return req_bandwidth_of_vnf * vnf_storage
        round_sto = req_bandwidth_of_vnf * vnf_storage
        return round(round_sto, 2)

    def CalculateCpuOfVnf(self, req_bandwidth_of_vnf, vnf_cpu):
        """
               Args:
               vnf_cpu (float): CPU required by a specific VNF in Mbps
               req_bandwidth_of_vnf (float): in Mbps
               """
        # return req_bandwidth_of_vnf * vnf_cpu
        round_cpu = req_bandwidth_of_vnf * vnf_cpu
        return round(round_cpu, 2)

    def CreatSliceType(self, req_vnf_numbers, vnf_cpu_sto_value_list, req_bandwidth_of_vnf):

        """
        Args:
        req_vnf_numbers (int):
        req_bandwidth_of_vnf (float): in Mbps
        """

        slice_vnfs = list()
        for element in req_vnf_numbers:
            vnf_storage = vnf_cpu_sto_value_list[element - 1][0]
            vnf_processing = vnf_cpu_sto_value_list[element - 1][1]
            # print("get VNF no. = ", element, " | ", vnf_storage, vnf_processing)

            if element == 3:
                req_bandwidth_of_vnf *= 0.8
            if element == 5:
                req_bandwidth_of_vnf *= 0.7
            if element == 6:
                req_bandwidth_of_vnf *= 0.6
            # print("req_bandwidth_of_vnf:", req_bandwidth_of_vnf)
            vnf_storage = self.CalculateStorageOfVnf(req_bandwidth_of_vnf,
                                                     vnf_storage)  ## return varribale = vnf_storage (float) is in GiB and rounded to 2 decimal places
            vnf_processing = self.CalculateCpuOfVnf(req_bandwidth_of_vnf,
                                                    vnf_processing)  ## return varribale = vnf_processing (float) is in number of CPU and rounded to 2 decimal places
            temp_tuple = (vnf_processing, vnf_storage)
            slice_vnfs.append(temp_tuple)

        # print("slice_vnfs:", slice_vnfs)
        # input()
        return slice_vnfs

    def getSliceTypeVnfDatarate(self, req_vnf_numbers, req_bandwidth_of_vnf):
        slice_vnfs_data_rate = list()

        for element in req_vnf_numbers:
            if element == 3:
                req_bandwidth_of_vnf *= 0.8
            if element == 5:
                req_bandwidth_of_vnf *= 0.7
            if element == 6:
                req_bandwidth_of_vnf *= 0.6

            slice_vnfs_data_rate.append(req_bandwidth_of_vnf)
        return slice_vnfs_data_rate

    def sliceVnfNames2VnfNumbers(self, vnf_name_list, req_vnf_names):
        slice_vnf_names_to_vnf_numbers = []
        for i in req_vnf_names:
            find_index = 0
            for j in vnf_name_list:
                find_index += 1
                if i == j:
                    slice_vnf_names_to_vnf_numbers.append(find_index)
        return slice_vnf_names_to_vnf_numbers

    def readConfigurationFromFile(self, file_name):
        ins = open(file_name, "r")
        data = []
        for line in ins:
            number_strings = line.split()  # Split the line on runs of whitespace
            numbers = [float(n) for n in number_strings]  # Convert to integers
            data.append(numbers)  # Add the "row" to your list.
        return data

    def getSliceTypeLatency(self, max_response_latency):  # in milliseconds:
        return int(random.uniform(*self.max_response_latency)), int(max_response_latency[1])

    def next_sfc_load(self):

        vnf_name_list = ['IDPS', 'FW', 'NAT', 'TM', 'VOC', 'WOC']
        vnf_cpu_sto_value_list = [[0.00166667, 0.000333333],
                                  [0.000833333, 0.000333333],
                                  [0.000333333, 0.000166667],
                                  [0.000333333, 0.000166667],
                                  [0.00333333, 0.000333333],
                                  [0.00166667, 0.0005]]

        req_vnf_numbers = self.sliceVnfNames2VnfNumbers(vnf_name_list, self.req_vnf_names)

        '''
        print("req_vnf_names:", self.req_vnf_names)
        print("req_vnf_numbers:", req_vnf_numbers)
        '''
        while True:
            sfc_params = {}

            req_latency, max_response_latency = self.getSliceTypeLatency(self.max_response_latency)  ## in ms
            per_user_datarate = self.GetSliceTypePerUserDatarate(self.per_user_data_rate)  ## in Mbps
            num_of_user_in_slice = self.GetSliceTypeAndNumberOfUsers(self.initial_number_of_users,  ## in int
                                                                     self.min_and_max_number_of_users)
            req_bandwidth_of_vnf = num_of_user_in_slice * per_user_datarate  ## in Mbps

            req_list_of_vnfs_with_cpu_and_sto = self.CreatSliceType(req_vnf_numbers, vnf_cpu_sto_value_list,
                                                                    req_bandwidth_of_vnf)  ## (cpu,sto)

            ### -------------- not yet used BEGINS --------------
            req_list_of_vnfs_with_data_rate = self.getSliceTypeVnfDatarate(req_vnf_numbers,
                                                                           req_bandwidth_of_vnf)  ## req_bandwidth_of_vnf (float) in Mbps

            delays = [0] * len(vnf_name_list)  # all vnf delays = 0
            ### -------------- not yet used ENDS --------------

            sfc_params['ingress'] = self.ingress
            sfc_params['egress'] = self.egress
            sfc_params['vnfs'] = req_list_of_vnfs_with_cpu_and_sto
            sfc_params['processing_delays'] = delays
            sfc_params['max_response_latency'] = req_latency  ## in ms

            ## req_bandwidth_of_vnf is in Mbps and the computing platform bandwidth and ICR is provided in Gbps
            ## convert Mbps to Gbps  ------  1.0 Mbps == 0.001 Gbps
            ## divide the value of req_bandwidth_of_vnf in Mbps by 1000 to get the value of vnf_storage in Gbps

            ### Experimental: reduce b/w by 10 folds
            sfc_params['bandwidth_demand'] = round(((req_bandwidth_of_vnf / 1000) / 10), 7)  ## (float) in Gbps
            # sfc_params['bandwidth_demand'] = ((req_bandwidth_of_vnf / 1000) / 10)  ## (float) in Gbps

            # sfc_params['bandwidth_demand'] = round((req_bandwidth_of_vnf / 1000), 2)  ## (float) in Gbps
            sfc_params['per_user_datarate'] = per_user_datarate  ## in Mbps
            sfc_params['number_of_users'] = num_of_user_in_slice  ## in int

            yield sfc_params


def displayReqDetail(request_batch):
    for req in range(len(request_batch)):
        print("sfc.arrival_time: ", request_batch[req][1].arrival_time)
        print("sfc.ttl: ", request_batch[req][1].ttl)
        print("sfc.bandwidth_demand: ", request_batch[req][1].bandwidth_demand)
        print("sfc.num_vnfs: ", request_batch[req][1].num_vnfs)
        print("sfc.vnfs: ", request_batch[req][1].vnfs)
        print("sfc.max_response_latency: ", request_batch[req][1].max_response_latency)
        print("sfc.processing_delays: ", request_batch[req][1].processing_delays)


def displayReqDetail2(request_batch):
    print("\nSFC Requests\n")
    vnf_properties_1 = ["(vnf_cpu, vnf_storage)"] * request_batch[0][1].num_vnfs
    print(tabulate(request_batch, headers=["(arrival_time, sfc_num)", vnf_properties_1]))


def countNumberofRequests(text, file_name):
    results = pd.read_csv(file_name)  ## on_bad_lines='skip' not working
    print(text, len(results))


def countNumberofRequests2(text, file_name):
    with open(file_name, mode='rb') as read_file:
        num_lines = sum(1 for line in read_file)
        print(text, num_lines)


def createRequestFile(file_name, request_batch):
    file_name = os.path.splitext(file_name)[0]
    file_name += '.csv'

    # print("request file_name: ", file_name)
    with open(file_name, encoding='UTF8', mode='w', newline='') as write_file:
        writer = csv.writer(write_file)

        header = ['arrival_time', 'ttl', 'ingress', 'egress', 'number_of_users', 'per_user_data_rate',
                  'max_response_latency',
                  'bandwidth_demand', 'num_vnfs']
        vnf_properties_1 = ['vnf_cpu', 'vnf_storage'] * request_batch[0][1].num_vnfs
        vnf_properties_2 = ['vnf_proc_delays'] * request_batch[0][1].num_vnfs

        for stri in vnf_properties_1:
            header.append(stri)
        for stri in vnf_properties_2:
            header.append(stri)

        writer.writerow(header)
        # print(header)
        # input()

        for req in range(len(request_batch)):
            data = []

            data.append(request_batch[req][1].arrival_time)
            data.append(request_batch[req][1].ttl)
            data.append(request_batch[req][1].ingress)
            data.append(request_batch[req][1].egress)
            data.append(request_batch[req][1].number_of_users)
            data.append(request_batch[req][1].per_user_datarate)
            data.append(request_batch[req][1].max_response_latency)
            data.append(request_batch[req][1].bandwidth_demand)
            data.append(request_batch[req][1].num_vnfs)

            vnfs = request_batch[req][1].vnfs
            for index, tuple in enumerate(vnfs):
                data.append(tuple[0])
                data.append(tuple[1])

            procs_dely = request_batch[req][1].processing_delays
            for val in procs_dely:
                data.append(val)

            writer.writerow(data)

    return file_name


def createServiceFiles(total_services, path, service_file_name, service_file_format):
    list_of_service_parameter_files = []

    for service_number in range(1, total_services + 1):
        service_parameter_file_name = (path + service_file_name + str(service_number) + service_file_format)
        list_of_service_parameter_files.append(service_parameter_file_name)
    return list_of_service_parameter_files


def createRequestFiles(list_of_service_parameter_files):
    list_of_request_files = []
    for service_parameter_file_name in list_of_service_parameter_files:
        with open(Path(service_parameter_file_name), 'r') as read_file:
            arrival_config = json.load(read_file)
            # print(arrival_config)
        arrival_process = ArrivalProcess.factory(arrival_config)

        request_batch = arrival_process.requests
        request_file_name = createRequestFile(service_parameter_file_name, request_batch)
        list_of_request_files.append(request_file_name)

    return list_of_request_files


def readServiceFiles(total_services, service_file_name, service_file_format):
    list_of_service_parameter_files = []
    list_of_request_files = []

    for service_number in range(1, total_services + 1):
        service_parameter_file_name = (service_file_name + str(service_number) + service_file_format)
        # print("service_parameter_file_name: ", service_parameter_file_name)
        # input()
        with open(Path(service_parameter_file_name), 'r') as read_file:
            arrival_config = json.load(read_file)

        arrival_process = ArrivalProcess.factory(arrival_config)
        request_batch = arrival_process.requests

        list_of_service_parameter_files.append(service_parameter_file_name)
        request_file_name = createRequestFile(service_parameter_file_name, request_batch)
        list_of_request_files.append(request_file_name)
    return list_of_service_parameter_files, list_of_request_files


def mergeAllRequestFiles(all_requests, list_of_request_files):
    with open(all_requests, "w", newline='', encoding='utf-8') as write_file:
        for request_file_name in list_of_request_files:
            with open(request_file_name, "r", encoding='utf-8') as read_file:
                next(read_file)  # skip header line or the first line of each csv file
                contents = read_file.read()
                write_file.write(contents)


def sortRequestFile(request_file_sorted, request_file_unsorted, column_num):
    """
    Each requests is stored in a row.
    The first column of each row is arrival_time of request
    The Second column of each row is ttl of request
    Sorts the requests based on request arrival_time or ttl
    Args:
        str request_file_sorted: Name of the file in which the sorted results are saved
        str request_file_unsorted: Name of the file containing requests that are not sorted
        int column_num: 0 = arrival_time, 1= ttl

    """
    with open(request_file_unsorted, "r", encoding='utf-8') as read_file, open(request_file_sorted, "w", newline='',
                                                                               encoding='utf-8') as write_file:
        reader = csv.reader(read_file, delimiter=',')
        writer = csv.writer(write_file)
        ### reminder arrival_time is at column zero(0) and ttl is at column one(1) of each_row of data in a csv file
        # sort = sorted(reader, key=operator.itemgetter(0), reverse=False) ## sorts based on str values
        sort = sorted(reader, key=lambda ech_row: float(ech_row[column_num]),
                      reverse=False)  ## sorts based on float values
        for each_row in sort:
            writer.writerow(each_row)
            # print(each_row)


def removeDuplicatesFromRequestFile(request_file, request_file_sorted, column_num):
    """
       removes duplicate requests.
       Each requests is stored in a row.
       The first column of each row is arrival_time of request
       The Second column of each row is ttl of request
       Removes the duplicate requests based on request arrival_time or ttl
       Checks the fist or second column of each row with next row, if similar keeps the fist row and removes the second
       Args:
           request_file_sorted: Name of the file containing requests that are sorted ased on arrival_time
           request_file: Name of the file containing requests that are unique based on arrival_time
           int column_num: 0 = arrival_time, 1= ttl
       """
    with open(request_file_sorted, 'r', encoding='utf-8') as read_file, open(request_file, 'w', newline='',
                                                                             encoding='utf-8') as write_file:
        seen = set()  # set for fast O(1) amortized lookup
        ## arrival_time is at column zero(0) of each_row of data in a csv file
        csv_data = csv.reader(read_file, delimiter=',')
        writer = csv.writer(write_file)
        for each_row in csv_data:
            # print(each_row[arrival_time])
            if each_row[column_num] in seen:
                continue  # skip duplicate
            seen.add(each_row[column_num])
            writer.writerow(each_row)


def readRequestsFromFileForInput(request_file_sorted):
    """
       Reads SFC requests from a sorted CSV file and returns a list of ServiceFunctionChain objects.

       Parameters:
       - request_file_sorted (str): Path to the sorted CSV file.

       Returns:
       - requests (list): List of ServiceFunctionChain objects.
       - num_requests (int): Number of requests in the list.
       """

    print("readRequestsFromFileForInput()")
    requests = []

    with open(request_file_sorted, 'r', encoding='utf-8') as read_file:
        csv_data = csv.reader(read_file, delimiter=',')
        for each_row in csv_data:
            # print(each_row)
            arrival_time = int(each_row[0])
            ttl = int(each_row[1])
            ingress = str(each_row[2])
            egress = str(each_row[3])
            number_of_users = int(each_row[4])
            per_user_datarate = float(each_row[5])
            max_response_latency = float(each_row[6])
            bandwidth_demand = float(each_row[7])
            num_vnfs = int(each_row[8])  ### read the number of VNFs this request has at the 8th column in csv file.
            num_vnfs = num_vnfs * 2  ### since each vnf has two properties (cpu and sto) we multiply num_vnfs by 2
            x = (num_vnfs + 8)  ### add the current index of the column to get the index of last vnf sto in the csv file
            # x is the index of last vnf sto in the csv file
            vnf_cpu_sto = []
            for num_vnfs in range(9, x + 1):
                vnf_cpu_sto.append(float(each_row[num_vnfs]))  ### get all vnfs
            it = iter(vnf_cpu_sto)
            vnfs = list(zip(it, it))  ### convert the list of vnf float values to tuple
            # print(vnfs)
            # input()

            sfc = ServiceFunctionChain(ingress, egress, arrival_time, ttl, bandwidth_demand, max_response_latency, vnfs,
                                       number_of_users, per_user_datarate, processing_delays=None, vnf_datarate=None)
            requests.append(sfc)

    return requests, len(requests)


def runRequestCreation():
    path = "parameters\\"
    total_services = 4
    service_file_name = "service"
    service_file_format = ".json"

    list_of_service_parameter_files = createServiceFiles(total_services, path, service_file_name, service_file_format)

    list_of_request_files = createRequestFiles(list_of_service_parameter_files)

    request_file_merged = path + 'all_requests_merged.csv'
    mergeAllRequestFiles(request_file_merged, list_of_request_files)

    """
    sort the file 'all_requests_merged.csv' based on ttl and save results in 'all_requests_sorted_ttl.csv'
    remove duplicate ttl values  
    """
    column_num = 1  ## ttl
    request_file_sorted_ttl = path + 'all_requests_sorted_ttl.csv'
    sortRequestFile(request_file_sorted_ttl, request_file_merged, column_num)

    request_file_ttl = path + 'all_requests_ttl.csv'
    removeDuplicatesFromRequestFile(request_file_ttl, request_file_sorted_ttl, column_num)

    column_num = 0  ## arrival_time
    request_file_sorted = path + 'all_requests_sorted.csv'
    sortRequestFile(request_file_sorted, request_file_ttl, column_num)

    request_file = path + 'all_requests.csv'
    removeDuplicatesFromRequestFile(request_file, request_file_sorted, column_num)

    countNumberofRequests2("requests default: ", request_file_merged)
    # countNumberofRequests2("requests sorted: ", request_file_sorted)
    countNumberofRequests2("requests final: ", request_file)

    ## Move extra request files to another folder for debugging and viewing purposes
    os.makedirs("parameters\debug_service", exist_ok=True)
    move_to_path = path + "debug_service\\"
    os.replace(request_file_merged, move_to_path + 'all_requests_merged.csv')
    os.replace(request_file_sorted_ttl, move_to_path + 'all_requests_sorted_ttl.csv')
    os.replace(request_file_ttl, move_to_path + 'all_requests_ttl.csv')
    os.replace(request_file_sorted, move_to_path + 'all_requests_sorted.csv')

    ### Experimental: generate only n number of requests
    # write_file_name = path + 'all_requests.csv'
    # number_of_requests = 2
    # keepNrowsFromCsvFile(number_of_requests, request_file, write_file_name)

    return request_file


### -----------------------Experimental code---------------------------------------------------
def keepNrowsFromCsvFile(rows_to_keep, read_file_name, write_file_name):
    df_firstn = pd.read_csv(read_file_name, header=None, nrows=rows_to_keep)
    df_firstn.to_csv(write_file_name, header=False, index=False)


### --------------------------------------------------------------------------


if __name__ == '__main__':
    request_file = runRequestCreation()
    requests = readRequestsFromFileForInput(request_file)
