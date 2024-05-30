# UERL

Source code for an adaptive mitigation method designed to address uncorrected errors (UEs) in DRAM. Leveraging Reinforcement Learning (RL) techniques, the method dynamically adapts to the probability and potential cost of encountering such errors, offering a proactive approach to mitigate their impact.

This is the supplemental code for the HPDC24 paper ["Reinforcement Learning-based Adaptive Mitigation of Uncorrected DRAM Errors in the Field"]() (url comming soon).

The UERL code is released under the BSD-3 [License](LICENSE).


## Code status

This repository is currently undergoing cleanup and optimization in preparation for the upcoming HPDC24 conference publication. We are actively working on improving its readability, performance, and documentation to ensure its usability for the wider community.

Our current focus is on generating useful synthetic logs for job samples and features of our RL model.


## Running scripts

Our project includes two main scripts: run.py and evaluate_best.py. Below is a guide on how to use these scripts to train and evaluate your reinforcement learning (RL) agents.

### Main Scripts

- run.py: This script is used to train the RL agent and evaluate its performance on the validation set.
- evaluate_best.py: This script takes the best performing agents from the validation set and evaluates them on the test data.

### Configuration File

Before running the scripts, ensure that your configuration file is properly set up. The configuration file contains all the necessary parameters for training and evaluation. An example configuration file can be found at [config/sample.yaml](config/sample.yaml).

### Running run.py

To start training and validating the RL agent, use the run.py script. You can specify the configuration file, the split to be used for training, the output directory, and additional options like verbose mode. Here is the command to run run.py:

    `python run.py --config config/sample.yaml --split 0 --output evaluations/validation/ -v`


- `--config config/sample.yaml`: Specifies the path to the configuration file.
- `--split 0`: Specifies the data split to be used for training.
- `--output evaluations/validation/`: Specifies the directory where the validation results will be saved.
- `-v`: Enables verbose mode for more detailed output.

You can check all the available parameters and options in the [Arguments](##Arguments) section.

**Note**: If hyperparameters are not specified in the configuration file, they will be randomly set during training.

### Running evaluate_best.py

After training multiple agents on different splits, you can use the evaluate_best.py script to evaluate the best performing agents from the validation set on the test data. Here is the command to run evaluate_best.py:

    `python evaluate_best.py --config config/sample.yaml --input evaluations/validation/ --output evaluations/test/test_sample.pkl -v`

`--config config/sample.yaml`: Specifies the path to the configuration file. If must be the same one used for training the agents.
`--input evaluations/validation/`: Specifies the directory containing the validation results.
`--output evaluations/test/test_sample.pkl`: Specifies the file where the test evaluation results will be saved.
`-v`: Enables verbose mode for more detailed output.

You can check all the available parameters and options in the [Arguments](##Arguments) section.

The resulting test_sample.pkl file is a pickle you can open in any Python script. For example:

    ```py
    import pickle

    with open('evaluations/test/test_sample.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data)
    ```

This file is a dictionary summarizing the evaluation on test data in the cross-validation. It contains the following information:

- `test_results`: Contains detailed test results for each data split. This typically includes performance metrics, such as UE and migitation costs, and other relevant data points specific to each test instance.
- `avg_ue_cost_per_split`: Average UE cost for each data split.
- `avg_mitigation_cost_per_split`: Average cost of mitigation actions taken for each data split.
- `avg_cost_per_split`: Average total cost per data split, combining both UE cost and mitigation cost. This provides an overall measure of performance and cost-efficiency for each data split.
- `total_avg_ue_cost`: Total average UE cost across all data splits. This is a cumulative measure of the impact on UE across the entire dataset.
- `total_avg_mitigation_cost`: Total average mitigation cost across all splits. This represents the cumulative resources or costs expended on mitigation efforts across the entire dataset.
- `total_avg_cost`: Total average cost across all data , combining both UE cost and mitigation cost. This provides an overall measure of performance and cost-efficiency across the entire dataset.

## Arguments

### run.py

Arguments for the `run.py` script for training and evaluating an RL agent on a specific data splitu sing a provided configuration YAML file. The evaluation is conducted on validation data.

| Short Option | Long Option | Description | Required |
|--------|-------------|-------------|----------|
| -s     | --split   | Number specifying the ith split to be used for training. | Yes |
| -c     | --config    | Path to configuration YAML file. | Yes |
| -o     | --output    | Path to output. Can be either a directory or a pickle file. If not specified, it will default to `evaluations/validation/validation_agent_{agent_id}.pkl`. | No |
| -v     | --verbose   | Enable verbose mode. | No |
| -d     | --debug     | Enable debug mode. | No |


### evaluate_best.py

Arguments for the `evaluate_best.py` script used to calculate the performance of the best validation agent on test data for each data split.

| Short Option | Long Option | Description | Required |
|--------|-------------|-------------|----------|
| -c     | --config    | Path to configuration YAML file. | Yes |
| -i     | --input     | Path to input directory with the validation files. | Yes |
| -o     | --output    | Path to output. Can be either a directory or a pickle file. If not specified, it will default to evaluations/test/test.pkl. | No |
| -v     | --verbose   | Enable verbose mode. | No |
| -d     | --debug     | Enable debug mode. | No |


# Acknowledgment

The work was supported by the Spanish Government, under the contracts PID2019-107255GB-C21 and CEX2021-001148-S funded by MCIN/AEI/ 10.13039/501100011033. The work also received funding from the Department of Research and Universities of the Government of Catalonia to the AccMem Research Group (Code: 2021 SGR 00807). Paul Carpenter holds the Ramon y Cajal fellowship RYC2018-025628-I funded by MICIU/AEI/10.13039/501100011033 and "ESF Investing in your future".