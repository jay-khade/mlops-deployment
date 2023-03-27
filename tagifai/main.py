# --------------------------------- Importing Libraries -----------------------------

import pandas as pd
from pathlib import Path
import json
from typing import Dict
from argparse import Namespace
from tagifai import data, train, utils
from config import config
import mlflow
from numpyencoder import NumpyEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback

import warnings

warnings.filterwarnings("ignore")


# ------------------- Defining core operations --------------------

# import data
def elt_data():
    """ Extract, Load and Transform our dataset"""

    # Extract + Load
    # importing data and  labels
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)

    # saving data to folder
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # transform, merge datasets
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # drop rows with no tags
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)


# train model
def train_model(args_fp="config/args.json"):
    """Train a model given arguments"""

    # load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))


# optimization
def optimize(
        args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """Optimize hyperparameters.
    Args:
        args_fp (str): location of args.
        study_name (str): name of optimization study.
        num_trials (int): number of trials to run in study.
    """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    args = {**args.__dict__, **study.best_trial.params}
    utils.save_dict(d=args, filepath=args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyper parameters: {json.dumps(study.best_trial.params, indent=2)}")


# ------------------------------------ calling function for main.py -----------------------------

if __name__ == "__main__":
    elt_data()
