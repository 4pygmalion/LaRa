from setuptools import find_packages, setup

setup(
    name="lara",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=["mlflow", "scikit-learn"],
)
