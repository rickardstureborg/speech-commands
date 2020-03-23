# speech-commands
Final project for the Natural Language Processing Course at Northeastern

## Setting Up Virtual Env
Create the environment:
`conda env create --file=environment.yml`

Update the Environment:
`conda env update --name speech-commands --file environment.yml --prune`

## Running model
1. Set hyperparameters, dataset location and other settings in config.cfg

2. Set up virtual environment

3. To train model on dataset (UNIX command):
`./main.py --train --config_path config.cfg`
