import os
import csv
import copy
from typing import Dict
import numpy as np
import pandas as pd
from collections import defaultdict

class solution_sfc():
    def __init__(self, ingress, egress, arrival_time, ttl, bandwidth_demand, max_response_latency, vnfs,
                 number_of_users=None,
                 per_user_datarate=None, processing_delays=None, vnf_datarate=None):

        self.arrival_time = arrival_time
        self.ttl = ttl
        self.bandwidth_demand = bandwidth_demand
        self.max_response_latency = max_response_latency
        self.vnfs = vnfs
        self.num_vnfs = len(self.vnfs)
        self.number_of_users = number_of_users
        self.per_user_datarate = per_user_datarate

        self.processing_delays = [0 for _ in self.vnfs] if processing_delays is None else processing_delays
        self.vnf_datarate = [0 for _ in self.vnfs] if vnf_datarate is None else vnf_datarate
        self.ingress = ingress
        self.egress = egress


class network():
    def __init__(self, ingress, egress, arrival_time, ttl, bandwidth_demand, max_response_latency, vnfs,
                 number_of_users=None,
                 per_user_datarate=None, processing_delays=None, vnf_datarate=None):
        self.arrival_time = arrival_time
        self.ttl = ttl
        self.bandwidth_demand = bandwidth_demand
        self.max_response_latency = max_response_latency
        self.vnfs = vnfs
        self.num_vnfs = len(self.vnfs)
        self.number_of_users = number_of_users
        self.per_user_datarate = per_user_datarate

        self.processing_delays = [0 for _ in self.vnfs] if processing_delays is None else processing_delays
        self.vnf_datarate = [0 for _ in self.vnfs] if vnf_datarate is None else vnf_datarate
        self.ingress = ingress
        self.egress = egress