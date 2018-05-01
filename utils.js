// Utility functions

// Convert a tf.Tensor to a nested array
// Tested for 2-D array
function nestedArrayFrom2dTensor(tensor) {
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

