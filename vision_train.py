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

from azureml.core.authentication import ServicePrincipalAuthentication, MsiAuthentication

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


def train(args):

    # Get current run and experiment
    run = Run.get_context()
    if (isinstance(run, azureml.core.run._OfflineRun)):
        ws = Workspace.from_config()
        experiment_name = 'automl-vision'
    else:
        experiment_name = run.experiment.name
        
        workspace_name = run.experiment.workspace.name
        subscription_id = run.experiment.workspace.subscription_id
        resource_group = run.experiment.workspace.resource_group
        
        try:
            print("Trying to authenticate to workspace using MSI")
            auth = MsiAuthentication()
            ws = Workspace(subscription_id=subscription_id,
                           resource_group=resource_group,
                           workspace_name=workspace_name,
                           auth=auth)
        except:
            print("Trying to authenticate to workspace using SP")
            ws_kv = run.experiment.workspace
            keyvault = ws_kv.get_default_keyvault()
            tenant_id = keyvault.get_secret(name="automl-tenant-id")
            sp_id = keyvault.get_secret(name="automl-service-principal-id")
            sp_password = keyvault.get_secret(name="automl-service-principal-password")
            
            auth = ServicePrincipalAuthentication(tenant_id=tenant_id,
                                                  service_principal_id=sp_id,
                                                  service_principal_password=sp_password)
            ws = Workspace(subscription_id=subscription_id,
                           resource_group=resource_group,
                           workspace_name=workspace_name,
                           auth=auth)

    # Re-create run with fully-authenticated workspace object
    experiment = Experiment(ws, name=experiment_name)
    
    print(f"Retrieved access to workspace {ws}")
    print(f"Experiment for logging: {experiment}")
        
    # Reference Training/Validation Dataset
    training_dataset = ws.datasets.get(args.training_dataset)
    print("Training dataset name: " + training_dataset.name)
   
    if args.validation_dataset:
        validation_dataset = ws.datasets.get(args.validation_dataset)
        print("Validation dataset name: " + validation_dataset.name)
        
    # TODO: Figure out how to check if Dataset is of right type
        
    # Configure Models
    if (args.models == 'all' and args.task == 'image-classification'):
        models_to_try = choice(["resnet50", "resnet18", "mobilenetv2", "seresnext"])
    elif (args.models == 'all' and args.task == 'image-object-detection'):
        models_to_try = choice(["fasterrcnn_resnet50_fpn", "fasterrcnn_resnet18_fpn", "yolov5"])
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
        
        general_settings = {
            'task': args.task,
            'compute_target': args.compute_cluster,
            'training_data': training_dataset
        }
        
        try:
            default_data['validation_data'] = validation_dataset
        except NameError:
            print("Skipping validation dataset...")
        
        automl_vision_config = AutoMLVisionConfig(**general_settings, **tuning_settings)
        
        # Need to submit as new experiment, child run can't be of type AutoML
        automl_vision_run = experiment.submit(automl_vision_config)       
        automl_vision_run.wait_for_completion(wait_post_processing=True)

        # Generate summary
        run = next(automl_vision_run.get_children())
        hdrun = HyperDriveRun(run.experiment, run.id)
        hdruns = hdrun.get_children_sorted_by_primary_metric()
        print(f"Hyperdrive Runs: {hdruns}")
        df = pd.DataFrame.from_dict(data = hdruns)
        results_df = df.dropna()

        # Write summary back
        print("Writing results summary back...")
        os.makedirs(args.results, exist_ok=True)
        save_data_frame_to_directory(args.results, results_df)
    except Exception as e:
        raise

if __name__ == '__main__':
    args = parse_args()
    train(args)