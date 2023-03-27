# --------------------------------- Importing Libraries -----------------------------

import pandas as pd
from pathlib import Path
import json
from argparse import Namespace
from tagifai import data,train,utils
from config import config

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


# ------------------------------------ calling function for main.py -----------------------------

if __name__ == "__main__":
    elt_data()
