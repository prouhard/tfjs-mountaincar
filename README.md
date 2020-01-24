# TensorFlow.js Example: Reinforcement Learning with Mountain Car Simulation

## Overview

This is a modification of [this tensorflow.js subrepository](https://github.com/tensorflow/tfjs-examples/tree/master/cart-pole), by [@Caisq](https://github.com/caisq), to tackle the OpenAI's gym MountainCar problem:
  https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py


For a more graphical illustration of the problem, see:
  http://gym.openai.com/envs/MountainCar-v0/

### Features:

- Allows user to specify the architecture of the policy network, in particular,
  the number of the neural networks's layers and their sizes (# of units).
- Allows training of the policy network in the browser, optionally with
  simultaneous visualization of the cart-pole system.
- Allows testing in the browser, with visualization.
- Allows saving the policy network to the browser's IndexedDB. The saved policy
  network can later be loaded back for testing and/or further training.

## Usage

```sh
yarn && yarn watch
```
