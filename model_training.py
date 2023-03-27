from config import config
from tagifai import main
from pathlib import Path

# args_fp = Path(config.CONFIG_DIR, "args.json")
# main.train_model()


main.train_model(experiment_name="baselines", run_name="sgd")

