import { MountainCar } from './mountain_car';
import { Model } from './model';
import { Memory } from './memory';
import { maybeRenderDuringTraining, setUpUI } from './ui';

import * as tf from '@tensorflow/tfjs';

const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.2;
const LAMBDA = 0.01;

export class Orchestrator {
    /**
     * @param {MountainCar} mountainCar
     * @param {Model} model
     * @param {Memory} memory
     * @param {number} discountRate
     * @param {number} maxStepsPerGame
     */
    constructor(mountainCar, model, memory, discountRate, maxStepsPerGame) {
        // The main components of the environment
        this.mountainCar = mountainCar;
        this.model = model;
        this.memory = memory;

        // The exploration parameter
        this.eps = MAX_EPSILON;

        // Keep tracking of the elapsed steps
        this.steps = 0;
        this.maxStepsPerGame = maxStepsPerGame;

        this.discountRate = discountRate;

        // Initialization of the rewards and max positions containers
        this.rewardStore = new Array();
        this.maxPositionStore = new Array();
    }

    /**
     * @param {number} position
     * @returns {number} Reward corresponding to the position
     */
    computeReward(position) {
        let reward = 0;
        if (position >= 0) {
            reward = 5;
        } else if (position >= 0.1) {
            reward = 10;
        } else if (position >= 0.25) {
            reward = 20;
        } else if (position >= 0.5) {
            reward = 100;
        }
        return reward;
    }

    async run() {
        this.mountainCar.setRandomState();
        let state = this.mountainCar.getStateTensor();
        let totalReward = 0;
        let maxPosition = -100;
        let step = 0;
        while (step < this.maxStepsPerGame) {

            // Rendering in the browser
            await maybeRenderDuringTraining(this.mountainCar);

            // Interaction with the environment
            const action = this.model.chooseAction(state, this.eps);
            const done = this.mountainCar.update(action);
            const reward = this.computeReward(this.mountainCar.position);

            let nextState = this.mountainCar.getStateTensor();

            // Keep the car on max position if reached
            if (this.mountainCar.position > maxPosition) maxPosition = this.mountainCar.position;
            if (done) nextState = null;

            this.memory.addSample([state, action, reward, nextState]);

            this.steps += 1;
            // Exponentially decay the exploration parameter
            this.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);

            state = nextState;
            totalReward += reward;
            step += 1;
            
            // Keep track of the max position reached and store the total reward
            if (done || step == this.maxStepsPerGame) {
                this.rewardStore.push(totalReward);
                this.maxPositionStore.push(maxPosition);
                break;
            }
        }
        await this.replay()
    }

    async replay() {
        // Sample from memory
        const batch = this.memory.sample(this.model.batchSize);
        const states = batch.map(([state, , , ]) => state);
        const nextStates = batch.map(
            ([, , , nextState]) => nextState ? nextState : tf.zeros([this.model.numStates])
        );
        // Predict the values of each action at each state
        const qsa = states.map((state) => this.model.predict(state));
        // Predict the values of each action at each next state
        const qsad = nextStates.map((nextState) => this.model.predict(nextState));

        let x = new Array();
        let y = new Array();

        // Update the states rewards with the discounted next states rewards
        batch.forEach(
            ([state, action, reward, nextState], index) => {
                const currentQ = qsa[index];
                currentQ[action] = nextState ? reward + this.discountRate * qsad[index].max().dataSync() : reward;
                x.push(state.dataSync());
                y.push(currentQ.dataSync());
            }
        );

        // Clean unused tensors
        qsa.forEach((state) => state.dispose());
        qsad.forEach((state) => state.dispose());

        // Reshape the batches to be fed to the network
        x = tf.tensor2d(x, [x.length, this.model.numStates])
        y = tf.tensor2d(y, [y.length, this.model.numActions])

        // Learn the Q(s, a) values given associated discounted rewards
        await this.model.train(x, y);

        x.dispose();
        y.dispose();
    }
}
