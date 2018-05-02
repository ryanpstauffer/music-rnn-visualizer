// Utility functions

// Convert a tf.Tensor to a nested array
// Tested for 2-D array
export function nestedArrayFrom2dTensor(tensor) {
  const shape = tensor.shape;
  const flatData = tensor.dataSync();
  
  let nestedArray = [];
  for (let i = 0; i < shape[0]; i++) {
    let innerArray = [];
    for (let j = 0; j < shape[1]; j++) {
      innerArray.push(flatData[i+j]);
    }
    nestedArray.push(innerArray);
  }

  return nestedArray; 
}


export function getParameters(rnn, test=false) {
  const params = {}
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


const pitchMapping = {
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

export function pitchNumberToNote(pitchNum) {
  return pitchMapping[pitchNum];
}


