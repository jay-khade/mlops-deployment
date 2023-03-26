# ---------------------------------------- Importing Libraries --------------------------------------

from pathlib import Path
from setuptools import find_namespace_packages, setup

# --------------------------- Instructions o how to set up virtual environment ---------------------

# extracting required packages from requirements.txt
BASE_DIR = Path(__file__).parent

with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# setup.py
setup(
    name='tagifai',
    version=0.1,
    description='classify machine learning projects',
    author='Goku Mohandas',
    python_requires='>=3.7',
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)
