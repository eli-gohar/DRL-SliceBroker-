# DRL-SliceBroker

## Overview

The code provided here represents an experimental implementation during the initial exploration into deep reinforcement learning (DRL). Subsequently, this code evolved to become the foundation for the experiments detailed in the paper titled ["An online cost minimization of the slice broker based on deep reinforcement learning"](https://doi.org/10.1016/j.comnet.2024.110198), which has been accepted for publication in [Computer Networks](https://www.sciencedirect.com/journal/computer-networks). 

## Project Purpose

The primary objective of this project was to address the online cost minimization of slice broker through the application of DRL using various RL algorithms.

## Disclaimer

***This code is provided as-is, and not the final version of the code for the paper.*** 

## Requirements

Conda was used to create the environment, and the requirements are specified in the environment.yml file.

## Usage

The command-line arguments:

```--agent PPO --network parameters\graph.gpickle --requests parameters\service1.json  --logs 1_tensorboard_logs\ --output 2_results\```

## Structure

Describe the organization of the repository, outlining key directories and their purposes.

## `src` Directory
Contains the source code files.

### `agent` Subdirectory
This directory contains files related to the agent functionality.

- `baselines.py`: Implements baseline algorithms.
- `logging.py`: Handles logging functionalities.
- `recorder.py`: Manages recording functionalities.

### `environment` Subdirectory
This directory contains files related to the environment.

- `env.py`: Based on `gym` describes the environment where the reinforcement learning (RL) agent operates.
- `inp_topology.py`: Implements the underlaying infrastructure provider (InP) topology.
- `network.py`: Manages network-related functionalities.
- `nsr.py`: Creates network slice requests (NSRs).
- `sfc.py`: Implements service function chaining (SFC)


## Acknowledgments

The code base [Computer Networks group @ UPB](https://github.com/CN-UPB/NFVdeep/tree/main), served as a foundational reference during the early stages of learning to impliment DRL.


## License

DRL-SliceBroker © 2022 by Ali Gohar is licensed under CC BY 4.0 
