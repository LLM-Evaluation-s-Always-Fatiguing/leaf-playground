from setuptools import setup, Extension, find_packages

version = "0.1.0.dev0"

requirements = [
    "pydantic>=2.0.0",
    "datasets",
    "openai>=1.0.0",
    "torch>=2.0.0",
    "transformers[agent]",
    "sentencepiece",
    "accelerate",
    "fastapi[all]"
]

setup(
    name="leaf_playground",
    version=version,
    install_requires=requirements,
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8.0"
)
