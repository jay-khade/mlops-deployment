# --------------------------- importing Libraries ----------------------------

from pathlib import Path
import pretty_errors

# ------------------ define key directory locations -------------------

# Assets
PROJECTS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, 'config')
DATA_DIR = Path(BASE_DIR,"data")

# create directory
DATA_DIR.mkdir(parents=True, exist_ok=True)
