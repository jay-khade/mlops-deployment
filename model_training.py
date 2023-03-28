from config import config
from tagifai import main
from pathlib import Path

# args_fp = Path(config.CONFIG_DIR, "args.json")
# main.train_model()

#main.train_model(experiment_name="baselines", run_name="sgd")

# text = "Transfer learning with transformers for text classification."
text = "Text Classification & Sentiment Analysis tutorial"
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
main.predict_tag(text=text, run_id=run_id)
