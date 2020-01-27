import * as tf from '@tensorflow/tfjs';


export class Model {
    /**
     * @param {number} numStates
     * @param {number} numActions
     * @param {number} batchSize
     */

    constructor(hiddenLayerSizesOrModel, numStates, numActions, batchSize) {
      this.numStates = numStates;
      this.numActions = numActions;
      this.batchSize = batchSize;

      if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
        this.network = hiddenLayerSizesOrModel;
        this.network.summary();
        this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
     } else {
        this.defineModel(hiddenLayerSizesOrModel);
      }
    }

    defineModel(hiddenLayerSizes) {

        if (!Array.isArray(hiddenLayerSizes)) {
            hiddenLayerSizes = [hiddenLayerSizes];
        }
        this.network = tf.sequential();
        hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
        this.network.add(tf.layers.dense({
            units: hiddenLayerSize,
            activation: 'relu',
            // `inputShape` is required only for the first layer.
            inputShape: i === 0 ? [this.numStates] : undefined
            }));
        });
        this.network.add(tf.layers.dense({units: this.numActions}));

        this.network.summary();
        this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    }

    /**
     * @param {tf.Tensor | tf.Tensor[]} states
     * @returns {tf.Tensor | tf.Tensor} The predictions of the best actions
     */
    predict(states) {
        return tf.tidy(() => this.network.predict(states));
    }

    /**
     * @param {tf.Tensor[]} xBatch
     * @param {tf.Tensor[]} yBatch
     */
    async train(xBatch, yBatch) {
        await this.network.fit(xBatch, yBatch);
    }

    /**
     * @param {tf.Tensor} state
     * @returns {number} The action chosen by the model (-1 | 0 | 1)
     */
    chooseAction(state, eps) {
        if (Math.random() < eps) {
            return Math.floor(Math.random() * this.numActions) - 1;
        } else {
            return tf.tidy(() => {
                const logits = this.network.predict(state);
                const sigmoid = tf.sigmoid(logits);
                const probs = tf.div(sigmoid, tf.sum(sigmoid));
                return tf.multinomial(probs, 1).dataSync()[0] - 1;
            });
        }
    }
}
