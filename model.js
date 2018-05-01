// Forget-All Gated RNN model
// Model definition

class ForgetAllGatedRNN {
  constructor(internalStateSize, outputVectorSize, learningRate) {
    this.internalStateSize = internalStateSize;
    this.outputVectorSize = outputVectorSize;
    this.learningRate = tf.scalar(learningRate);

    // Setup
    this.time = 0;
    this.numTrainingUpdates = 0;
    this.lastOutput = tf.variable(tf.zeros([this.outputVectorSize, 1]));

    this.initialize();
    this.loopLength = 1;
  }

  initialize() {
    // Perform initializations for our RNN
    // Initialize cell internal state
    this.internalState = tf.variable(tf.zeros([this.internalStateSize, 1]));
  
    // Initialize Weight matrics
    this.W = tf.variable(tf.randomNormal(
        [this.internalStateSize, this.internalStateSize], 0, 0.01));
 
    this.U = tf.variable(tf.randomNormal(
        [this.internalStateSize, this.outputVectorSize], 0, 0.01));

    this.V = tf.variable(tf.randomNormal(
        [this.outputVectorSize, this.internalStateSize], 0, 0.01)); 

    // Initialize bias vectors
    this.b = tf.variable(tf.zeros([this.internalStateSize, 1]));
    this.c = tf.variable(tf.zeros([this.outputVectorSize, 1]));
 
    // Initialize intermediate matrices and vectors for visualization
    this.a = tf.variable(tf.zeros([this.internalStateSize, 1]));
    this.forgetGate = tf.variable(tf.zeros([this.internalStateSize, 1]));
  }

  step() {
    this.updateInternalState_();
    const logits = this.generateLogits_();
    // tf.softmax() only support max on last dimension
    // This causes failure on [3,1] shaped vector!
    // We tranpose and retranpose as a work around
    const probs = logits.transpose().softmax().transpose();

    // Prepare for next step
    this.lastOutput = tf.oneHot(probs.argMax(), this.outputVectorSize)
                        .reshape([this.outputVectorSize, 1]); 
    // Note that our time counter records the timestep of the
    // last completed output
    this.time++;

    return probs;
  } 

  // Generate values from the network
  // TODO: placeholder for delivery of values to viz
  generate(numSteps) {
    for (let n = 0; n < numSteps; n++) {
      let probs = this.step(); 
      let output = probs.argMax().dataSync();
      console.log(output);
    }
  }
 
  getLastOutputAsPitch() {
    return pitchNumberToNote(this.lastOutput.argMax().dataSync()[0] + 60);
  }

  trainEpoch(targetOutputs, targetProb=0.98) {
    if (this.loopLength == 1) {
      this.loopLength = targetOutputs.length;
    }

    let unrolled = {
      internalStates: {},
      probs: {},
      lastOutputs: {}
    };
    let p = null;
    let t = 0;

    // for (let i = 0; i < 4; i++) {
    for (let i = 0; i < this.loopLength; i++) {
      p = this.step();
      // t = (this.time - 1) % 4 + 1;
      t = (this.time - 1) % this.loopLength + 1;
      unrolled.probs[t] = p.clone();
      unrolled.internalStates[t] = this.internalState.clone();
      unrolled.lastOutputs[t] = this.lastOutput.clone();
    }
      // After 4 timesteps, we've generated enough outputs
      // to match targets
      // So let's backprop and update!
    let evalProgress = {}; 
    //if (this.time % 4 == 0) {
    if (this.time % this.loopLength == 0) {
      console.log('TRAINING');
      let gradients = this.calcGradients_(
        unrolled.internalStates, unrolled.lastOutputs,
        unrolled.probs, targetOutputs);
      this.updateParameters_(gradients); 
      this.numTrainingUpdates++;

      // Evaluate
      evalProgress.loss = gradients.loss.clone().dataSync()[0];
      console.log(
        'Current Loss, time ' + this.time + ': ' + evalProgress.loss)

      const correctProbsBuf = tf.buffer([targetOutputs.length, 1]);

      // TODO: generalize for multiple loop lengths
      // for (let n = 0; n < 4; n++) {
      for (let n = 0; n < this.loopLength; n++) {
        correctProbsBuf.set(unrolled.probs[n+1].get(targetOutputs[n], 0), n, 0);
      }
      let correctProbs = correctProbsBuf.toTensor();
      evalProgress.correctProbs = correctProbs.clone().dataSync();
       
      // console.log(evalProgress.correctProbs);
      evalProgress.minCorrectProbability = Math.min(...evalProgress.correctProbs);
      console.log('Minimum correct probability: ', 
                  evalProgress.minCorrectProbability);
       
      evalProgress.correct = evalProgress.correctProbs.map(x => x > targetProb);
      // console.log(evalProgress.correct);
      evalProgress.numCorrect = evalProgress.correct.reduce((a, b) => a + b, 0);
      // console.log('Number correct: ', evalProgress.numCorrect);
    }
    return evalProgress; 
  } 
  // Train the network cell until targetProbability is reached.
  // targetOutput: a vector with length that matches loop length.
  trainToCompletion(targetOutputs, targetProb=0.98) {
    let evalProgress = {};

    let minCorrectProbability = 0;
    let correct = [];
    // We set a ceiling on iterations, to avoid non-terminating loops
    // in cases of non-convergence
    while (minCorrectProbability < targetProb && this.time < 5000) {
      evalProgress = this.trainEpoch(targetOutputs, targetProb);
      minCorrectProbability = evalProgress.minCorrectProbability;
    }
  }

  calcGradients_(internalStates, lastOutputs, probs, targetOutputs) {
    // Forward pass
    // Target outputs right now have to have the indices adjusted to work!
    const loss = tf.variable(tf.scalar(0));
    for (let t = 1; t < this.loopLength + 1; t++) {
      loss.assign(
        loss.sub(tf.log(tf.scalar(probs[t].get(targetOutputs[t-1], 0)))));
    } 

    // Setup
    const d_W = tf.variable(tf.zerosLike(this.W));
    const d_U = tf.variable(tf.zerosLike(this.U));
    const d_V = tf.variable(tf.zerosLike(this.V));
    const d_b = tf.variable(tf.zerosLike(this.b));
    const d_c = tf.variable(tf.zerosLike(this.c));
    const d_logits = tf.variable(tf.zerosLike(probs[1]));

    const d_nextState = tf.variable(tf.zerosLike(this.internalState));

    //for (let t = 4; t > 0; t--) {
    for (let t = this.loopLength; t > 0; t--) {
      // The gradient w.r.t. output function
      d_logits.assign(
        probs[t].sub(tf.oneHot(
            tf.scalar(targetOutputs[t-1], 'int32'),
            this.outputVectorSize)
          .reshape([this.outputVectorSize, 1])));

      // Gradients in output function
      // y_t = V * h_t + c
      d_V.assign(
        d_V.add(
          d_logits.matMul(internalStates[t].transpose())));

      d_c.assign(d_c.add(d_logits));

      // Since the state update lies behind the forget gate at t=1,
      // We can ignore this part of the error propagation if t=1
      if (t > 1) {
        // Gradients in state update function
        // h_t = W * h_t-1 + U * y_t-1 + b
        let d_state = tf.matMul(this.V.transpose(), d_logits)
                        .add(d_nextState);

        let d_a = tf.ones([this.internalStateSize, 1])
                    .sub(internalStates[t].square())
                    .mul(d_state);

        d_W.assign(
          d_W.add(
            d_a.matMul(internalStates[t-1].transpose())));

        d_U.assign(
          d_U.add(
            d_a.matMul(lastOutputs[t-1].transpose())));

        d_b.assign(d_b.add(d_a));
      
        d_nextState.assign(
          tf.matMul(this.W.transpose(), d_a));
      }
    }
 
    // Finalize gradients object 
    let gradients = {
      loss: loss,
      d_W: d_W,
      d_U: d_U,
      d_V: d_V,
      d_b: d_b,
      d_c: d_c,
      d_logits: d_logits
    }

    return gradients;
  }

  // Update parameters
  updateParameters_(gradients) {
    this.W.assign(
      this.W.sub(this.learningRate.mul(gradients.d_W)));
    this.U.assign(
      this.U.sub(this.learningRate.mul(gradients.d_U)));
    this.V.assign(
      this.V.sub(this.learningRate.mul(gradients.d_V)));
    this.b.assign(
      this.b.sub(this.learningRate.mul(gradients.d_b)));
    this.c.assign(
      this.c.sub(this.learningRate.mul(gradients.d_c)));
  }


  updateInternalState_() {
    this.a.assign(this.W.matMul(this.internalState).add(
                this.U.matMul(this.lastOutput)).add(
                this.b));
    if (this.time % 4 == 0) {
      this.forgetGate.assign(tf.fill([this.internalStateSize, 1], 0));
    } else {
      this.forgetGate.assign(tf.fill([this.internalStateSize, 1], 1));
    }
    // const a = this.W.matMul(this.internalState).add(
    //             this.U.matMul(this.lastOutput)).add(
    //             this.b);
    // const forgetGate = (this.time % this.loopLength == 0) ? tf.scalar(0) : tf.scalar(1);
    // const forgetGate = (this.time % 4 == 0) ? tf.scalar(0) : tf.scalar(1);
    // if (forgetGate.dataSync() == 0 ) { console.log('FORGET'); } 
    this.internalState.assign(this.a.tanh().mul(this.forgetGate));
  }

  generateLogits_() {
    return tf.matMul(this.V, this.internalState).add(this.c);
  }

  printParameters() {
    console.log('W:', this.W.toString());
    console.log('U:', this.U.toString());
    console.log('V:', this.V.toString());
    console.log('b:', this.b.toString());
    console.log('c:', this.c.toString()); 
  }
}

