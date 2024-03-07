class ServiceFunctionChain:

    def __init__(self, ingress, egress, arrival_time, ttl, bandwidth_demand, max_response_latency, vnfs,
                 number_of_users=None,
                 per_user_datarate=None, processing_delays=None, vnf_datarate=None):
        '''Creating a new service function chain

        Args:
          arrival_time (int): Arrival time of the SFC
          ttl (int): Time To Live of the SFC
          bandwidth_demand (float): The maximum ammount of bandwidth required by SFC
          max_response_latency (float): The maximal acceptable latency
          vnfs (list of tuples): A (ordered) list of all Vnfs of the SFC i.e., vnfs[(cpu,storage)]
          number_of_users (int): number of users supported by SFC
          per_user_data_rate (float): Datarate required by one user
          processing_delays (list of float): A (ordered) list of all delays of the Vnfs (default: no delays)
          vnf_datarate (list of float): A (ordered) list of all datarates of the Vnfs (default: no delays)'''

        '''Description of SFC arguments
          
          arrival_time (int): Based on Possion process
          ttl (int): Based on deletion timeslot
          bandwidth_demand (float): Based on  (number_of_users * per_user_data_rate) 
          vnfs (list of tuples): Based on (number_of_users) 
          vnf_datarate (list of float): Based on  (number_of_users * per_user_data_rate)
        '''

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

    def __repr__(self):
        """String representation of the SFC instance."""
        s = ' '.join([str([vnf for vnf in self.vnfs])])
        return s
