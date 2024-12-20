# Decision Support Systems Using Greedy Algorithm
Decision Support System (DSS) for multiclass classification tasks, leveraging Conformal Prediction Sets and a novel Greedy Optimization Algorithm. The system dynamically adapts to varying noise levels in real-world data, enhancing prediction accuracy and efficiency. Key innovations include counterfactual analysis for optimal prediction set selection and robust handling of noisy datasets. This project significantly improved human-AI collaboration by providing structured, reliable, and scalable decision support, validated through extensive experiments on ImageNet datasets with high noise variability.

## Requirements

This code was tested using Python 3.8 on a Linux system. To run the experiments, you'll need MPI installed on your machine. You can find instructions for installing MPI [here](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html).
If you want to generate the plots shown in the manuscript, make sure you have Latex installed.
To set up the required packages, use a [conda](https://conda.io/projects/conda/en/latest/index.html) environment on Linux system and run the following commands:

```bash
conda create --name hai-psets python=3.8
conda activate hai-psets
pip install -r requirements.txt
```

## Evaluation

In order to execute scripts, ensure you are in the base directory and have the correct environment activated.

```bash
cd decision-support-system-using-prediction-sets
conda activate hai-psets
export PYTHONPATH=.
```

Here is a bash script that runs all the experiments and analysis:
- `run_all_real_data.sh`: execute the real data experiment described in Section 8
- `generale_all_results.sh`: generate the Latex code for the various tables presented in the paper.

### Running experiments with real data

The script `run_real_experiment.py` executes the experiments using real data. For instance, the following command produces the results for the experiment discussed in Section 8.
```bash
mpirun - n 2 python run_real_experiment.py --model-epochs epoch10 --calibrate top-k --ranks 5 --calibration-size 800
```

## Code Structure

The directories have the following content:
- `data/`: it includes the ImageNet-16H dataset required for the experiments, along with Python utility functions for preprocessing.
- `results/`: it includes Python scripts for parsing and analyzing the results presented in the paper.
- `utils/`: it includes utility functions and the code needed to run the experiments.