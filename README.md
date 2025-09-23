# RL-Game
The sb3_* files are the meat of this project. The test* files are very early iterations and experiments that aren't relavent now.

Run sb3_train.py to train a connect4 model that plays against a random opponent. 
Specify a loaded AI to play against in sb3_play.py and then run it to play against the trained AI.

Running sb3_train_selfplay.py will train an AI that periodically freezes itself and updates the opponent AI to make training more effective.
(sb3_play.py will also work with this trained model so long as the file is specified)
