<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>Snake Game with DQN-based AI
</h1>
<h3>◦ Developed with the software and tools listed below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style&logo=NumPy&logoColor=white" alt="NumPy" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
</p>
<img src="https://img.shields.io/github/languages/top/simje71380/SnakeGameAI?style&color=5D6D7E" alt="GitHub top language" />
<img src="https://img.shields.io/github/languages/code-size/simje71380/SnakeGameAI?style&color=5D6D7E" alt="GitHub code size in bytes" />
<img src="https://img.shields.io/github/commit-activity/m/simje71380/SnakeGameAI?style&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/license/simje71380/SnakeGameAI?style&color=5D6D7E" alt="GitHub license" />
</div>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
  - [✔️ Prerequisites](#️-prerequisites)
  - [💻 Installation](#-installation)
  - [🎮 Using SnakeGameAI ](#-using-snakegameai)
- [📄 License](#-license)
---


## 📍 Overview

This repository contains an implementation of the classic Snake game where both humans and AI can play. The AI player is powered by the Deep Q-Network (DQN) algorithm. What sets this implementation apart is the flexibility it offers for configuring the AI's observation capabilities. There are four observation modes available:

1-Tile Vision: The AI observes only the tiles adjacent to its head. (in front, right, left)<br />
2-Tile Vision: The AI observes the two tiles surrounding its head.<br />
3-Tile Vision: The AI observes the three tiles around its head.<br />
Full Vision: The AI has complete visibility of the entire game board.

![](https://github.com/simje71380/SnakeGameAI/blob/main/snakeAI.gif)

---

## 🧩 Modules

<details open><summary>Root</summary>

| File                                                                             | Summary                                |
| ---                                                                              | ---                                    |
| [agent.py](https://github.com/simje71380/SnakeGameAI/blob/main/agent.py)         | Implementation of the Agent            |
| [helper.py](https://github.com/simje71380/SnakeGameAI/blob/main/helper.py)       | Display information during training    |
| [main.py](https://github.com/simje71380/SnakeGameAI/blob/main/main.py)           | main file : AI or keyboard play        |
| [model.py](https://github.com/simje71380/SnakeGameAI/blob/main/model.py)         | Model of the Neural Network            |
| [SnakeGame.py](https://github.com/simje71380/SnakeGameAI/blob/main/SnakeGame.py) | The SnakeGame implemented from scratch |
| [train.py](https://github.com/simje71380/SnakeGameAI/blob/main/train.py)         | Used to train the AI                   |

</details>

---

## 🚀 Getting Started

### ✔️ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
> - `ℹ️ Requirement 1`
> - `ℹ️ Requirement 2`
> - `ℹ️ ...`

### 📦 Installation

1. Clone the SnakeGameAI repository:
```sh
git clone https://github.com/simje71380/SnakeGameAI
```

2. Change to the project directory:
```sh
cd SnakeGameAI
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🎮 Using SnakeGameAI

<h4>To play as a human</h4>
Simply use the arrow keys to navigate the snake. The objective is to eat the food and grow the snake without colliding with the walls or itself.

```sh
python main.py human
```

<h4>To watch the AI play the game using the trained model:</h4>

For 1-Tile Vision: observation_mode = "1_tile"<br />
For 2-Tile Vision: observation_mode = "2_tile"<br />
For 3-Tile Vision: observation_mode = "3_tile"<br />
For Full Vision: observation_mode = "full"<br />

```sh
python main.py AI observation_mode
```
Example :
```sh
python main.py AI 1_tile
```

Note: the AI of observation_mode = "full" is still "dumb" and scores 0. This is due to the high number of inputs and low training time.


<h4>To train the AI using the DQN algorithm by running:</h4>

For 1-Tile Vision: observation_mode = "1_tile"<br />
For 2-Tile Vision: observation_mode = "2_tile"<br />
For 3-Tile Vision: observation_mode = "3_tile"<br />
For Full Vision: observation_mode = "full"<br />

```sh
python train.py observation_mode
```
Example:
```sh
python train.py 1_tile
```

## 📄 License
This project is licensed under the MIT License, which means you are free to use, modify, and distribute the code for your own purposes. See the [LICENSE](./LICENSE) file for additional info