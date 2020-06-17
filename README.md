# AI-Gamer

This project was aimed at writing algorithms able to learn how to play various games (for example Atari).

![](pong.gif)
![](breakout.gif)
![](pacman.gif)

[![](https://imgur.com/biAiR9L.png)](https://www.youtube.com/watch?v=eeM2Rdbufco)
## Installing Atari package under Windows
The ```openai\atari-py``` package is not officially supported under Windows. However some slightly modified versions make it compatible. To install the correct version, first check that there is currently no versions installed with ```pip show atari-py```. If a version different than ```1.2.1``` is installed, do ```pip uninstall atari-py```. Then the right version can be installed with : 
```
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

## How to use this repository
