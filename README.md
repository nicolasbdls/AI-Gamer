# AI-Gamer

This project was aimed at writing algorithms able to learn how to play various games (for example Atari). The games' environments are taken from OpenAI Gym. Below are shown some results that have been obtained:

![](pong.gif)
![](breakout.gif)
![](pacman.gif)

[![](https://imgur.com/biAiR9L.png)](https://www.youtube.com/watch?v=eeM2Rdbufco)
## Installing Atari package under Windows
The ```openai\atari-py``` package is not officially supported under Windows. However some slightly modified versions make it compatible. To install the correct version, first check that there is currently no versions installed with ```pip show atari-py```. If a version different than ```1.2.1``` is installed, do ```pip uninstall atari-py```. Then the right version can be installed with : 
```
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

## How to use this repository ?
This repository contains both python files (```.py```) and python notebooks (```.ipynb```). Indeed some algorithms have been written on Google Colab and offer the most features (for example the use of tensorboard), however, for convenience it is possible to run these codes on any interpreter with the python files.

The last version of  the DQN algorithm for Atari is presented in a single version applied to the game Pong.In can easily be applied to other games by just changing the name of the environment in ```gym.make()```. However, at the moment no satisfactory trained models for breakout or pacman have been obtained for a question of time. In order to observe good results for these two games, old versions of the DQN which were more specific to each game have been uploaded with the corresponding trained models. They can be found in "[Old but trained](https://github.com/nicolasbdls/AI-Gamer/tree/master/DQN/From%20pixels/Atari/Old%20but%20trained)" folder.

For the DQN applied to Atari, the "train" algorithms can be used to start training the agent or to coninue training from a pretrained model (in this case set epsilon to 0.02 to directly exploit the policy). The "render" algorithms can be used to visualize the results from a trained model and test the agent. It can be useful to observe the true score of the game as the printed score during training may be the score achieved within one life (see ```EpisodicLife``` wrapper) or with a clipped reward (see ```ClipReward```wrapper).

To run the DQN algorithms with convolutional layers, a GPU is necessary. Also a large RAM is necessary, most of the DQN were run using the "Big RAM" feature of Google Colab (25 GB).

## Associated report
The final report giving a full description of this project can be found [here](https://documentcloud.adobe.com/link/track?uri=urn:aaid:scds:US:cecbce1b-0d7c-4f68-8257-e124e598ae63).
