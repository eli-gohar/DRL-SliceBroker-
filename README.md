# DRL-SliceBroker

## Overview

This repository contains the code used in the experiments conducted for the paper titled ["An online cost minimization of the slice broker based on deep reinforcement learning"](https://doi.org/10.1016/j.comnet.2024.110198), which has been accepted for publication in [Computer Networks](https://www.sciencedirect.com/journal/computer-networks). The code provided here represents an experimental implementation that evolved from an initial exploration into deep reinforcement learning. 

## Disclaimer

**This code is provided as-is, and not the final version of the code for the paper. It was inspired by the code base [Computer Networks group @ UPB](https://github.com/CN-UPB/NFVdeep/tree/main), which served as a foundational reference during the early stages of development.**

## Requirements

Conda was used to creat the envioronment and requirments are presented in  `environment.yml`

## Usage

The command-line arguments:

```--agent PPO --network parameters\graph.gpickle --requests parameters\service1.json  --logs 1_tensorboard_logs\ --output 2_results\```

## Structure

Describe the organization of the repository, outlining key directories and their purposes.

- `src`: Contains the source code files.
- `data`: Placeholder for datasets used in experiments.
- `models`: Placeholder for saved model checkpoints.

## Contributions

If you welcome contributions, provide guidelines for how others can contribute to the project. Include details about the preferred process for submitting issues, proposing changes, and making pull requests.

## Acknowledgments

Acknowledge any external libraries, code bases, or resources that significantly influenced this work.

## License

Specify the license under which your code is released. If you're unsure, you can use an open-source license like MIT or Apache 2.0.

## Contact

Provide your contact information or a way for users to reach out with questions, feedback, or issues.
