// Music RNN Visualization Main module

import { ForgetAllGatedRNN } from './model.js';
import { MelodyScore, TargetScore } from './score.js';
import { RNNVisualization } from './visualization.js';
import { pitchNumberToNote } from './utils.js';

// User Interface
// Build Model
let internalStateSize;
let outputVectorSize = 13;
let learningRate = 0.5;

const hiddenLayerSlider = document.getElementById('hiddenLayerSlider');
const hiddenLayerValue = document.getElementById('hiddenLayerValue');
hiddenLayerValue.innerHTML = hiddenLayerSlider.value;

// Update slider value
hiddenLayerSlider.oninput = function() {
  hiddenLayerValue.innerHTML = this.value; 
}

let rnn, viz;

// const buildButton = document.getElementById('build');
// buildButton.addEventListener('click', function() {
document.getElementById('build')
    .addEventListener('click', function() {
      let internalStateSize = parseInt(hiddenLayerSlider.value); 
      // Remove existing Visualization elements
      d3.select('.networkStages').selectAll('*').remove();
      targetScore.clearScore();
      targetScore.loaded = false; 
      // Create new Model and Visualization 
      rnn = new ForgetAllGatedRNN(
          internalStateSize, outputVectorSize, learningRate);
      viz = new RNNVisualization(rnn);
      // Change button text
      this.innerHTML = 'Reset Model';
    }, false);


// ------ Model Training ------
// Note Entry

// Keyboard
// Define synth
let synth = new Tone.Synth().toMaster();
let targetMelody = [];

const score = new MelodyScore('score');

const keyboardContainer = document.querySelector('#inputKeyboard');
const keyboardInterface = new KeyboardElement(keyboardContainer); 

const targetScore = new TargetScore('targetScore');

d3.selectAll('.key')
    .on('mousedown', function() { 
      keyboardInterface.keyDown(this.id);
      synth.triggerAttack(pitchNumberToNote(this.id));
      // Add warning for targetMelody length?
      if (targetMelody.length < 8) {
        targetMelody.push(parseInt(this.id) - 60);
        score.addNoteToStaff(parseInt(this.id));
      }
            console.log(targetMelody);
    })
    .on('mouseup', function() {
      keyboardInterface.keyUp(this.id);
      synth.triggerRelease();
    });


const clearButton = document.getElementById('clearMelody');
clearButton.addEventListener('click', function() {
  targetMelody = [];
  score.clearScore();
  console.log(targetMelody);
 }, false);

let evalProgress = {minCorrectProbability: 0.0};
let targetProb = 0.98;
const autoLearnSlider = document.getElementById('autoLearnSlider');
const learnButton = document.getElementById('learn');

// Progress Bar
const currentLossValue = document.getElementById('currentLoss');
const numCorrectValue = document.getElementById('numCorrect');
const currentTimeStep = document.getElementById('currentTimeStep');
const currentNumUpdates = document.getElementById('currentNumUpdates');


// Update Progress bar and model visualization
function updateVisualization(evalProgress) {
    targetScore.recolorNotesFromProbs(evalProgress.correctProbs);
    console.log(evalProgress);
    currentLossValue.innerHTML = Math.round(evalProgress.loss * 1000) / 
                                 1000;
    numCorrectValue.innerHTML = evalProgress.numCorrect; 
    currentTimeStep.innerHTML = parseInt(rnn.time); 
    currentNumUpdates.innerHTML = parseInt(rnn.numTrainingUpdates);
    viz.showWeights();
}


learnButton.addEventListener('click', function() {
  // Ensure that the targetMelody is loaded in the score
  if (targetScore.loaded == false) {
    targetScore.addTargetMelodyToStaff(targetMelody);
  }
  if (parseInt(autoLearnSlider.value) == 0) {
    // "Step-through" mode
    // Execute a single step of training
    evalProgress = rnn.trainEpoch(targetMelody);
    updateVisualization(evalProgress);
    // targetScore.recolorNotesFromProbs(evalProgress.correctProbs);
    // console.log(evalProgress);
    // currentLossValue.innerHTML = Math.round(evalProgress.loss * 1000) / 
    //                              1000;
    // numCorrectValue.innerHTML = evalProgress.numCorrect; 
    // currentTimeStep.innerHTML = parseInt(rnn.time); 
    // currentNumUpdates.innerHTML = parseInt(rnn.numTrainingUpdates);
    // viz.showWeights();
  } else if (parseInt(autoLearnSlider.value) == 1) {
    // "Auto-learn" mode
    // Execute training to completion
    // We set a ceiling on iterations, to avoid non-terminating loops
    // in cases of non-convergence
    // TODO: Need this to re-render
    const delay = 10; // 100ms
    const maxIterations = 2000;
    function timeoutLoop() {
      evalProgress = rnn.trainEpoch(targetMelody, targetProb);
      updateVisualization(evalProgress);
      // currentLossValue.innerHTML = Math.round(evalProgress.loss * 1000) /
      //                              1000;
      // numCorrectValue.innerHTML = evalProgress.numCorrect; 
      // currentTimeStep.innerHTML = parseInt(rnn.time); 
      // currentNumUpdates.innerHTML = parseInt(rnn.numTrainingUpdates);
      // viz.showWeights();
      if (evalProgress.minCorrectProbability < targetProb &&
           rnn.time < maxIterations) {
        setTimeout(timeoutLoop, delay);
      } else if (rnn.time > maxIterations) {
        document.getElementById('warning').classList.toggle('hidden');
      }
    }

    setTimeout(timeoutLoop, delay);
    // while (evalProgress.minCorrectProbability < targetProb &&
    //        rnn.time < 2000) {
    //   evalProgress = rnn.trainEpoch(targetMelody, targetProb);

    //   // Update Progress bar and model visualization
    //   currentLossValue.innerHTML = Math.round(evalProgress.loss * 1000) /
    //                                1000;
    //   numCorrectValue.innerHTML = evalProgress.numCorrect; 
    //   currentTimeStep.innerHTML = parseInt(rnn.time); 
    //   currentNumUpdates.innerHTML = parseInt(rnn.numTrainingUpdates);
    //   viz.showWeights();
    // }
  }
}, false);

// Playback

// Update tempo value from slider
const tempo = document.getElementById('tempoSlider');
tempo.oninput = function() {
  Tone.Transport.bpm.value = parseInt(this.value); 
}

let pitch;

Tone.Transport.schedule(function(time) {
  pitch = viz.visualizeStep();
  console.log(pitch);
  synth.triggerAttackRelease(pitch, '4n');
}, 0)

// Set initial Tone Transport setting
Tone.Transport.bpm.value = tempo.value;
Tone.Transport.loopEnd = '4n';
Tone.Transport.loop = true;

const playButton = document.getElementById('playButton');
playButton.addEventListener('click', function() {
  this.classList.toggle('playing');
  if (this.classList.contains('playing')) {
    Tone.Transport.start();
    this.innerHTML = '&#9724';
  } else {
    Tone.Transport.stop();
    this.innerHTML = '&#x25b6';
  }
}, false);

// const topButton = document.getElementById('topRowButton');
// topButton.onclick(showRow(topButton.innerHTML));
