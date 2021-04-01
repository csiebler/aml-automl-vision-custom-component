import os
import argparse
import pandas as pd

from azureml.core import Workspace, Experiment, Run, Dataset, Run

import azureml.automl.core
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.train.hyperdrive import HyperDriveRun

from azureml.train.automl import AutoMLVisionConfig
from azureml.train.hyperdrive import GridParameterSampling, RandomParameterSampling, BayesianParameterSampling
from azureml.train.hyperdrive import BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, uniform

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

# Parse args
def parse_args():
    parser = argparse.ArgumentParser("AutoML-Scoring")
    parser.add_argument("--models", type=str, help="Models to try out")
    parser.add_argument("--task", type=str, help="Task type")
    parser.add_argument("--training_dataset", type=str, help="Training Dataset Name")
    parser.add_argument("--validation_dataset", type=str, help="Validation Dataset Name")
    parser.add_argument("--compute_cluster", type=str, help="Compute Cluster name")
    parser.add_argument("--iterations", type=int, help="Number of model training iterations")
    parser.add_argument("--number_nodes", type=int, help="Number of nodes to train on")
    parser.add_argument("--results", type=str, help="Results Directory")
    return parser.parse_args()


def predict(args):

    # Get AutoML run
    run = Run.get_context()
    if (isinstance(run, azureml.core.run._OfflineRun)):
        ws = Workspace.from_config()
        experiment_name = 'automl-vision' 
        experiment = Experiment(ws, name=experiment_name)
    else:
        ws = run.experiment.workspace
        experiment = run.experiment
    print(f"Retrieved access to workspace {ws}")
        
    # Reference Training Dataset
    training_dataset_name = args.training_dataset
    training_dataset = ws.datasets.get(training_dataset_name)
    print("Training dataset name: " + training_dataset.name)
    
    # TODO: Figure out how to check if Dataset is of right type
    
    # print("Training dataset type: " + training_dataset.label['type'])
    
    # if (training_dataset.label['type'] != "Classification"):
    #     print("TODO: FAIL")
    
    # Check if cluster is there
    cluster_name = args.compute_cluster
    clusters = ws.compute_targets
    if cluster_name in clusters and clusters[cluster_name].type == 'AmlCompute':
        print('Found existing compute target.')
        compute_target = clusters[cluster_name]
    else:
        print('TODO: FAIL...')

    # Configure Models
    if (args.models == 'all'):
        models_to_try = choice(["resnet50", "resnet18", "mobilenetv2", "seresnext"])
    else:
        models_to_try = choice(args.models)
    print(models_to_try)
       
    try:
        parameter_space = {
            'model_name': models_to_try
        }
        
        tuning_settings = {
            'iterations': args.iterations,
            'max_concurrent_iterations': args.number_nodes,
            'hyperparameter_sampling': RandomParameterSampling(parameter_space),
            'policy': BanditPolicy(evaluation_interval=2, slack_factor=0.2, delay_evaluation=6)
        }
                
        automl_vision_config = AutoMLVisionConfig(task=args.task,
                                                  compute_target=compute_target,
                                                  training_data=training_dataset,
                                                  **tuning_settings)
        automl_vision_run = experiment.submit(automl_vision_config)
        automl_vision_run.wait_for_completion(wait_post_processing=True)

        # Generate summary
        run = next(automl_vision_run.get_children())
        hdrun = HyperDriveRun(run.experiment, run.id)
        hdruns = hdrun.get_children_sorted_by_primary_metric()
        print(f"Hyperdrive Runs: {hdruns}")
        df = pd.DataFrame.from_dict(data = hdruns)
        results_df = df.dropna()

        # Write results back
        print("Writing results back...")
        os.makedirs(args.results, exist_ok=True)
        save_data_frame_to_directory(args.results, results_df)
    except Exception as e:
        raise


if __name__ == '__main__':
    args = parse_args()
    predict(args)