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
To install **leaf-playground** from the source, you need to clone the project by using `git clone`, then in your local `leaf-playground` directory, run:
```shell
pip install .
```

## Usage

### Start Server and Create a Task

To start the server that contains projects hosted in **leaf-playground-hub**, you need to first clone this project, then in the directory of your local **leaf-playground-hub**, using CLI command to start server with webui:
```shell
leaf-out start-server [--port PORT] [--ui_port UI_PORT]
```

By default, the backend service will run on port 8000, the UI service will run on port 3000, you can use `--port` and `--ui_port` options to use different ports respectively.

Below is a video demonstrates how to create and run a task that using MMLU dataset to evaluate LLM-based agents.

https://github.com/LLM-Evaluation-s-Always-Fatiguing/leaf-playground/assets/754493/0c980a97-1b7f-4884-bd85-fbdc60121ac8

## Maintainers

[@PanQiWei](https://github.com/panqiwei); [@Pandazki](https://github.com/pandazki).

## Roadmap

### The Framework

- [x] support human participant in the scenario simulation as a dynamic agent
- [x] running each scenario simulation task in a docker container
- [ ] refactor `ai_backend` to `llm_backend_tools` to remove some heavy dependencies
- [x] support manage task status(pause, restart, interrupt, etc.)
- [ ] support full task data persistence
- [ ] support streaming agents' responses

### The Hub

- [x] optimize scene flow of `who_is_the_spy` project and add metrics and evaluators
- [ ] create a new project to support using OpenAI [evals](https://github.com/openai/evals)
- [ ] create a new project to support using Microsoft [promptbench](https://github.com/microsoft/promptbench)
