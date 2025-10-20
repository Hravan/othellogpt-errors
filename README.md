# othellogpt-errors

(Repository based on https://github.com/likenneth/othello_world)

Analysis of errors made by othellogpt from the paper Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task by Li et al. While the authors of the original paper claim that OthelloGPT understands the game and uses understanding to make predictions, the goal of this simple analysis is to show that it only, not surprisingly, stochastically repeats what it was given in the training data.


## Result replication
### Download required data

Download and add the following files to their respective directories:

- [Battery Othello](https://drive.google.com/file/d/1SBraI6Xb5m2L4aithi0yGq803QQsXZLY/view?usp=drive_link) → `battery_othello/`
- [Othello Championship Data](https://drive.google.com/file/d/1tRTYvT0X63rMLzN0ZhudPGFjsUpiFPpL/view?usp=drive_link) → `data/othello_championship/`
- [Model Checkpoints](https://drive.google.com/file/d/1bccTOFW4CmsmYiRGDJ_y23tuVpu3NcSP/view?usp=drive_link) → `ckpts/`
