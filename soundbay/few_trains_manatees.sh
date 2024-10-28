#!/bin/bash
python train.py --config-name runs/Manatees
python train.py --config-name runs/Manatees model=defaults
