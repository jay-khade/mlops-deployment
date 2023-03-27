from config import config
from tagifai import main
from pathlib import Path

# args_fp = Path(config.CONFIG_DIR, "args.json")
# main.train_model()


# args_fp = Path(config.CONFIG_DIR, "args.json")
main.optimize(num_trials=20)

