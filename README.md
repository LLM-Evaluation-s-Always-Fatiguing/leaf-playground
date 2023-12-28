<h1>
  <p align="center">
    <strong>leaf-playground</strong>
  </p>
</h1>

## Introduction

**leaf-playground** is a "definition driven development" framework to build scenario simulation projects that human and LLM-based agents can participant in together to compete to or co-operate with each other. It is primarily designed to efficiently evaluate the performance of LLM-based agents at the action level in specific scenarios or tasks, but it also possesses enormous potential for LLM native applications, such as developing [a language-based game](https://github.com/LLM-Evaluation-s-Always-Fatiguing/leaf-playground-hub/tree/main/who_is_the_spy).

Apart from the framework itself, a bunch of CLI commands are provided to help developers speedup the process of building a scenario simulation project, and easily deploy a server with a [WEB UI](https://github.com/LLM-Evaluation-s-Always-Fatiguing/leaf-playground-webui) where users can create simulation tasks, manually and(or) automatically evaluate agents' performance, visualize the simulation process and evaluation results.

Below are sister projects of **leaf-playground**:
- [**leaf-playground-webui**](https://github.com/LLM-Evaluation-s-Always-Fatiguing/leaf-playground-webui): the implementation of the leaf-playground's WEB UI.
- [**leaf-playground-hub**](https://github.com/LLM-Evaluation-s-Always-Fatiguing/leaf-playground-hub): hosts our officially implemented scenario simulation projects.

## Installation

### Environment Setup

Make sure you have `Python` and `Node.js` installed on your computer, if not, you can set up the environment by following instructions:
- install `Python`: we recommend to use [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) to configure Python virtual environment.
- install `Node.js`: you can download and install Node.js from [Node.js official site](https://nodejs.org/en).

### Quick Install
**leaf-playground** has already been upload to pypi, thus you can use `pip` to quickly install:
```shell
pip install leaf-playground
```

### Install from source
To install **leaf-playground** from the source, you need to clone the project by using `git clone`, then in the `leaf-playground` directory, run:
```shell
pip install .
```

## Usage

### Start Server and Create a Task


## Maintainers

## License