// Interface
pitchMapping = {
  60: 'C4',
  61: 'Db4',
  62: 'D4',
  63: 'Eb4',
  64: 'E4',
  65: 'F4',
  66: 'Gb4',
  67: 'G4',
  68: 'Ab4',
  69: 'A4',
  70: 'Bb4',
  71: 'B4',
  72: 'C5'
}

function pitchNumberToNote(pitchNum) {
  return pitchMapping[pitchNum];
}

function getParameters(rnn, test=false) {
  params = {}
  if (test==true) {
    params.W = tf.randomNormal([4, 4], 0, 1);
    params.U = tf.randomNormal([4, 3], 0, 1);
    params.b = tf.randomNormal([4, 1], 0, 1);
    params.V = tf.randomNormal([3, 4], 0, 1);
    params.c = tf.randomNormal([3, 1], 0, 1);
  } else {
    params.W = rnn.W.clone(); 
    params.U = rnn.U.clone();
    params.b = rnn.b.clone();
    params.V = rnn.V.clone();
    params.c = rnn.c.clone();
    params.a = rnn.a.clone();
    params.forgetGate = rnn.forgetGate.clone();
  }

  return params;
}


// Main
let internalStateSize;
let outputVectorSize = 13;
let learningRate = 0.5;

let synth = new Tone.Synth().toMaster();

// TODO: add initial visualization of weights

const hiddenLayerSlider = document.getElementById('hiddenLayerSlider');
const hiddenLayerValue = document.getElementById('hiddenLayerValue');
hiddenLayerValue.innerHTML = hiddenLayerSlider.value;

// Update slider value
hiddenLayerSlider.oninput = function() {
  hiddenLayerValue.innerHTML = this.value; 
}

let rnn, viz;

const buildButton = document.getElementById('build');
buildButton.addEventListener('click', function() {
  let internalStateSize = Number(hiddenLayerSlider.value); 
  // let outputVectorSize = 13;
  // let learningRate = 0.5;
  // Remove existing viz elements
  d3.select('.networkStages').selectAll('*').remove();
  rnn = new ForgetAllGatedRNN(internalStateSize, outputVectorSize, learningRate);
  viz = new RNNVisualization(rnn);
}, false);


// Model Training
// Note Entry
// Keyboard
let targetMelody = [];

score = new MelodyScore('score');

const keyboardContainer = document.querySelector('#inputKeyboard');
const keyboardInterface = new KeyboardElement(keyboardContainer); 

targetScore = new TargetScore('targetScore');

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

learnButton.addEventListener('click', function() {
  // Ensure that the targetMelody is loaded in the score
  if (targetScore.loaded == false) {
    targetScore.addTargetMelodyToStaff(targetMelody);
  }
  if (parseInt(autoLearnSlider.value) == 0) {
    // "Step-through" mode
    // Execute a single step of training
    evalProgress = rnn.trainEpoch(targetMelody);
    targetScore.recolorNotesFromProbs(evalProgress.correctProbs);
    console.log(evalProgress);
    currentLossValue.innerHTML = Math.round(evalProgress.loss * 1000) / 1000;
    numCorrectValue.innerHTML = evalProgress.numCorrect; 
    currentTimeStep.innerHTML = parseInt(rnn.time); 
    currentNumUpdates.innerHTML = parseInt(rnn.numTrainingUpdates);
    viz.showWeights();
  } else if (parseInt(autoLearnSlider.value) == 1) {
    // "Auto-learn" mode
    // Execute training to completion
    // We set a ceiling on iterations, to avoid non-terminating loops
    // in cases of non-convergence
    // TODO: Need this to re-render
    while (evalProgress.minCorrectProbability < targetProb && rnn.time < 5000) {
      evalProgress = rnn.trainEpoch(targetMelody, targetProb);
      currentLossValue.innerHTML = Math.round(evalProgress.loss * 1000) / 1000;
      numCorrectValue.innerHTML = evalProgress.numCorrect; 
      currentTimeStep.innerHTML = parseInt(rnn.time); 
      currentNumUpdates.innerHTML = parseInt(rnn.numTrainingUpdates);
      viz.showWeights();
    }
  }
}, false);

// Playback

// Update tempo value from slider
const tempo = document.getElementById('tempoSlider');
tempo.oninput = function() {
  Tone.Transport.bpm.value = parseInt(this.value); 
}

Tone.Transport.schedule(function(time) {
  pitch = viz.visualizeStep();
  console.log(pitch);
  synth.triggerAttackRelease(pitch, '4n');
}, 0)

// Set initial Tone Transport setting
Tone.Transport.bpm.value = tempo.value;
Tone.Transport.loopEnd = '4n';
Tone.Transport.loop = true;

const play = document.getElementById('playToggle');
play.addEventListener('change', function(e) {
  if (e.target.checked) {
    Tone.Transport.start();
  } else {
    Tone.Transport.stop();
  }
}, false);

