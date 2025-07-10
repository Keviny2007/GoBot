# GoBot

GoBot is a Go board game implementation with AI capabilities, allowing you to play against deep learning-based bots or watch bots play against each other. The project includes a graphical user interface built with Pygame.

## Features

* Play against a trained deep learning AI
* Watch AI bots play against each other
* Multiple trained models available
* Graphical user interface for game interaction
* Command line options for customizing gameplay

## Getting Started

Prerequisites
* Python 3.10+
* PyTorch
* Pygame
* NumPy

Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Keviny2007/GoBot.git
    cd GoBot
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

## Usage

Playing Against a Bot

Use the `human_v_bot.py` script to play against a trained AI:
    ```bash
    python human_v_bot.py

Command line options:
* --board-size: Set the board size (default: 19)
* --bot-color: Choose the color for the bot ('black' or 'white', default: 'white')
* --model: Specify a trained model file to use
* --list-models: List all available trained models

Example:
    ```bash
    python human_v_bot.py --board-size 19 --bot-color black

Watching Bots Play Against Each Other (not as exciting)

Use the `bot_v_bot.py` script to watch two bots play:
    ```bash
    python bot_v_bot.py

Command line options:
* --black: Type of black player bot ('random' or 'dl', default: 'random')
* --white: Type of white player bot ('random' or 'dl', default: 'dl')
* --black-model: Model file for black player (if using dl)
* --white-model: Model file for white player (if using dl)
* --list-models: List available models and exit

Example:
    ```bash
    python bot_v_bot.py --black dl --white dl --black-model model1.pth --white-model model2.pth

## AI Models

The project includes several pre-trained models located in the `dl_human_generated/epochs` directory. You can select these models when playing against a bot or watching bots play against each other.

## Acknowledgments
* Based on the Deep Learning and the Game of Go book concepts
* Implemented with PyTorch for deep learning functionality (original booked uses an outdated version of TensorFlow)