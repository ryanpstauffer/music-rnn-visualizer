// Visualization class and support functions
import { nestedArrayFrom2dTensor, getParameters } from './utils.js';
 

export class RNNVisualization{
  constructor(rnn) {
    this.rnn = rnn;

    // Set starting size parameters
    this.totalWidth_ = 800;
    this.totalHeight_ = 500;
    this.sectionHeight_ = 300;
    this.titleMargin_ = 50;

    // These are the margins applied to each SVG
    this.margin = {top: 20, right: 20, bottom: 20, left:20};
    this.squareSize_ = 20;
    this.rows = [];
    this.initialSetup_();
    this.showInitialWeights();
  }


  initialSetup_() {
    // Prepare updateState container and title 
    this.networkStages = d3.select('.networkStages')
        .attr('width', this.totalWidth_)
        .attr('height', this.totalHeight_);

    const outputKeyboardContainer = document.querySelector('#outputKeyboard');
    this.outputKeyboard = new KeyboardElement(outputKeyboardContainer);
  }


  addLinearAlgebraRow(rowName) {
    const newRow = new NetworkRow(
        this.networkStages, rowName, 0, this.yPos, this.squareSize_,
        this.margin);
    this.rows.push(newRow); 

    return newRow;
  }


  showInitialWeights() {
    let params = getParameters(this.rnn);
    let hPrev = this.rnn.internalState.clone();
    let yPrev = this.rnn.lastOutput.clone();
    let hCurrent = this.rnn.internalState.clone();
    this.setupUpdateState(params, hPrev, yPrev);
    this.setupEmitOutput(params, hCurrent);
    
    //this.visualizeNetwork(params, hPrev, yPrev, hCurrent)
  }


  showWeights() {
    let params = getParameters(this.rnn);
    let hPrev = this.rnn.internalState.clone();
    let yPrev = this.rnn.lastOutput.clone();
    let hCurrent = this.rnn.internalState.clone();
    
    this.visualizeNetwork(params, hPrev, yPrev, hCurrent)
  }


  visualizeStep() {
    let params = getParameters(this.rnn);
    let hPrev = this.rnn.internalState.clone();
    let yPrev = this.rnn.lastOutput.clone();
    let probs = this.rnn.step()
    // hCurrent = tf.randomNormal([4, 1], 0, 1);
    let hCurrent = this.rnn.internalState.clone();
    
    this.visualizeNetwork(params, hPrev, yPrev, hCurrent)

    // Convert probs to pitchProbMapping
    let flatProbs = nestedArrayFrom2dTensor(probs);
    let pitchProbMapping = {};
    for (let i = 0; i < 13; i++) {
      pitchProbMapping[i+60] = flatProbs[i][0]; 
    }
    console.log(pitchProbMapping);
    this.updateKeyboardProbs(pitchProbMapping);

   // TODO: clean up the interaction bn the model, viz and synth
    return this.rnn.getLastOutputAsPitch();
  }

  visualizeNetwork(params, hPrev, yPrev, hCurrent) {
    // The number of padding squares in between heatmaps 
    let numPaddingSquares = 1;
    let xPos = 0;
    let yPos = 0;
    let pos = [];
    let plusSize = 50;
    let containerHeight = (this.rnn.internalStateSize + 2) *
        this.squareSize_;
    let containerWidth = Math.max(
        this.rnn.internalStateSize + 2,
        this.rnn.outputVectorSize + 2) * this.squareSize_;
    let yPlus = containerHeight / 2 - plusSize/2; 
    let xPlus = 100 / 2 - plusSize/2; 

    let totalHeight = this.titleMargin_ * 3;
    this.rows[0].updateHeatmaps([params.W, hPrev, yPrev, params.U, params.b]);
    this.rows[1].updateHeatmaps([params.a, params.forgetGate]);
    this.rows[2].updateHeatmaps([params.V, hCurrent, params.c]);
  }


  setupUpdateState(params, hPrev, yPrev) {
    // Weight matrix W
    let numPaddingSquares = 1;
    // let xInitial = pad(numPaddingSquares, 0); 
    // let yInitial = pad(numPaddingSquares, 0);
    let xPos = 0;
    let yPos = 0; 
    let pos = [];
    let plusSize = 50;
    let containerHeight = (this.rnn.internalStateSize + 2)  * this.squareSize_;
    let containerWidth = Math.max(
        this.rnn.internalStateSize + 2,
        this.rnn.outputVectorSize + 2) * this.squareSize_;
    let yPlus = containerHeight / 2 - plusSize/2; 
    let xPlus = 100 / 2 - plusSize/2; 

    // First row (Weights + bias) 
    const topRow = this.addLinearAlgebraRow('Update State: Affine Transformation');

    pos = topRow.addTensorHeatmap(params.W, 'W');

    pos[0] += this.squareSize_;
    pos = topRow.addTensorHeatmap(hPrev, 'hPrev', [pos[0], 0]);

    pos[0] += this.squareSize_;
    let plusYPos = (this.rnn.internalStateSize + 2)/2 * this.squareSize_ - plusSize/2;
    pos = topRow.addPlus(50, 10, [pos[0], plusYPos]);
    pos[0] += this.squareSize_;
 
    let yPrevStart = pos[0];
    pos = topRow.addTensorHeatmap(yPrev, 'yPrev', [yPrevStart, 0], true);

    pos[1] += this.squareSize_;
    pos = topRow.addTensorHeatmap(params.U, 'U', [yPrevStart, pos[1]]);
    let innerHeight = pos[1];

    // Reset y and continue;
    pos[0] += this.squareSize_;
    pos = topRow.addPlus(50, 10, [pos[0], plusYPos]);
    pos[0] += this.squareSize_;
    
    pos = topRow.addTensorHeatmap(params.b, 'b', [pos[0], 0]);

    let innerWidth = pos[0];
    topRow.updateDimensions(innerWidth, innerHeight);

    const nextRow = this.addLinearAlgebraRow('Update State: Activation');
    pos = [0,0];
    let ySymbol = this.rnn.internalStateSize * this.squareSize_ / 2;
    pos = nextRow.addText('tanh', [0, ySymbol + 8]);
    pos = nextRow.addTensorHeatmap(params.a, 'a', [pos[0], 0]);
    // nextRow.padX(numPaddingSquares);

    pos[0] += this.squareSize_;
    pos = nextRow.addHadamard(8, [pos[0], ySymbol - 8]);
    // nextRow.padX(numPaddingSquares);
    pos[0] += this.squareSize_;

    pos = nextRow.addTensorHeatmap(
        params.forgetGate, 'forgetGate', [pos[0], 0]);
    nextRow.updateDimensions(pos[0], pos[1]);
  }

  setupEmitOutput(params, hCurrent) { 
    // Emit Output
    // Reset xPos
    let numPaddingSquares = 1;
    let xPos = 0;
    let yPos = 0;
    let pos = [0, 0];
    // We technically don't have to recalc this!
    let containerHeight = Math.max(
        this.rnn.internalStateSize, this.rnn.outputVectorSize) * this.squareSize_;
    let plusSize = 50;
    let yPlus = containerHeight / 2 - plusSize/2; 
    let xPlus = 100 / 2 - plusSize/2; 
   
    // First row (Weights + bias) 
    const emitRow = this.addLinearAlgebraRow('Emit Output');

    pos = emitRow.addTensorHeatmap(params.V, 'V', pos, true);
    // emitRow.padX(numPaddingSquares);

    pos[0] += this.squareSize_;
    pos = emitRow.addTensorHeatmap(hCurrent, 'hCurrent', [pos[0], 0]);
    // emitRow.padY(numPaddingSquares);
   
    let innerWidth = pos[0]; 
    // emitRow.xPos = emitRow.xPos / 2;

    pos[1] += this.squareSize_;
    let plusXPos = (this.rnn.outputVectorSize + 2) *
                   this.squareSize_ / 2 - plusSize/2;
    pos = emitRow.addPlus(50, 10, [plusXPos, pos[1]]);
    // emitRow.padY(numPaddingSquares);

    emitRow.xPos = 0;
    pos[1] += this.squareSize_;
    pos = emitRow.addTensorHeatmap(params.c, 'c', [0, pos[1]], true);
    let innerHeight= pos[1];
    emitRow.updateDimensions(innerWidth, innerHeight);
 
  } 


  // Using a mapping of pitchNumber: prob to update
  // The keyboard visualization
  updateKeyboardProbs(pitchProbMapping) {
    let keyColor = '#fff';
    for (let n = 60; n < 73; n++) {
      keyColor = float2grayscale(pitchProbMapping[n]); 
      this.outputKeyboard.keys[n].children[0].style.backgroundColor = keyColor;
    }
  }

}


// Visualization
function float2grayscale(percentage) {
  let colorPartDec = 255 * Number(percentage).toFixed(3);
  let colorPartHex = Number(parseInt(colorPartDec, 10)).toString(16);
  return '#' + colorPartHex + colorPartHex + colorPartHex;
}

const scaleWeightToColor = d3.scaleLinear()
    .range(['#70161e', '#f9f9f8', '#022f40'])
    .domain([-2, 0, 2]);

function updateMatrix(array2d, svgElement) {
  let newColor;
  let row = svgElement.selectAll('.row')
      .data(array2d);
  
  let square = row.selectAll('.square')
      .data(function(d) { return d; })
      // Update old data
      // Stand-in for testing
      .style('fill', function(d) { 
        // console.log('updating fill', d);
        // newColor = scaleWeightToColor(d);
        // console.log(newColor);
        // return newColor; });
        return scaleWeightToColor(d); });
}

function visualizeMatrix(array2d, svgElement, squareSize=30) {
  let row = svgElement.selectAll('.row')
      .data(array2d)
    .enter().append('g')
      .attr('class', 'row')
      .attr('transform', function(d, i) {
        return 'translate(0,' + (1 + i * squareSize) + ')';
      }); 
  
  let square = row.selectAll('.square')
      .data(function(d) { return d; })
      // Update old data
      .style('fill', function(d) { 
    //   return float2grayscale(d) });
        return scaleWeightToColor(d) })
    .enter().append('rect')
      .attr('class', 'square')
      .attr('x', function(d, i) { return 1 + i * squareSize })
      .attr('width', squareSize)
      .attr('height', squareSize)
      .style('fill', function(d) { 
    //   return float2grayscale(d) });
        return scaleWeightToColor(d) });
}

// Visualize output Probabilities 
// Places the viz within a pre-defined svg element
function visualizeOutputProbs(array2d, svgElement, squareSize=30) {
  let row = svgElement.selectAll('.row')
      .data(array2d)
    .enter().append('g')
      .attr('class', 'row')
      .attr('transform', function(d, i) {
        return 'translate(0,' + (1 + i * squareSize) + ')';
      }); 
  
  let square = row.selectAll('.square')
      .data(function(d) { return d; })
    .enter().append('rect')
      .attr('class', 'square')
      .attr('x', function(d, i) { return 1 + i * squareSize })
      .attr('width', squareSize)
      .attr('height', squareSize)
      .style('fill', function(d) { 
         return float2grayscale(d) });
}


// Create a named svg element for our heatmap
// className: string
// xPos: Integer
// yPos: int
function createSvgElement(className, containerElem, xPos=0, yPos=0) {
  let newElem = containerElem.append('g')
      .attr('class', className)
      .attr('transform', function(d) { 
          return 'translate(' + xPos + ',' + yPos + ')';
        });

  return newElem;
}


class TensorHeatmap {
  constructor(tensorName, transpose=false) {
    this.tensorName = tensorName;
    this.transpose = transpose;  
  }
} 


// Returns position of bottom right corner
function visualizeTensorHeatmap(
    tensor, tensorName, containerElem, xPos, yPos, squareSize=30,
    transpose=false) {
  // console.log('Visualizing tensor: ', tensorName, tensor);
  let nestedArray = [];
  let pos = [];
  if (transpose) {
    nestedArray = nestedArrayFrom2dTensor(tensor.transpose());
    pos = [xPos + tensor.shape[0] * squareSize,
           yPos + tensor.shape[1] * squareSize];
 } else {
    nestedArray = nestedArrayFrom2dTensor(tensor);
    pos = [xPos + tensor.shape[1] * squareSize,
           yPos + tensor.shape[0] * squareSize];
  }
  let svgElem = createSvgElement(tensorName, containerElem, xPos, yPos); 
  visualizeMatrix(nestedArray, svgElem, squareSize);
  
  // clean up padding and square size params
  return pos;
}


class NetworkRow {
  constructor(containerElem, id='', xStart=0, yStart=0, squareSize=20,
      margin={top: 0, right: 0, bottom:0, left:0}) {
    this.containerElem = containerElem;
    this.xPos = 0;
    this.yPos = 0;
    this.innerWidth = 0;
    this.innerHeight = 0;
    this.outerWidth = margin.right + margin.left;
    this.outerHeight = margin.top + margin.bottom;
    this.squareSize_ = squareSize;
    this.elements = [];
    this.heatmaps = [];
    this.elementsConsumingTensors = [];
    this.margin = margin;

    this.button = this.containerElem.append('button')
        .attr('class', 'accordion')
        .attr('id', id + 'Button')
        .html(id)
        .on('click', toggleShowRow);

    this.svg = this.containerElem.append('svg')
        .attr('class', 'networkRow')
        .attr('id', id)
        .attr('transform', 'translate(' + xStart + ',' + yStart + ')');
    this.g = this.svg.append('g')
        .attr('transform', 'translate(' + this.margin.left + ',' +
              this.margin.top + ')');

  }

  updateDimensions(
      innerWidth=this.innerWidth, innerHeight=this.innerHeight) {
    this.outerWidth = innerWidth + this.margin.right + this.margin.left;
    this.outerHeight = innerHeight + this.margin.top + this.margin.bottom;
    this.g.attr('width', innerWidth).attr('height', innerHeight);
    this.svg.attr('width', this.outerWidth).attr('height', this.outerHeight);
  }

  updateHeatmaps(tensors) {
    let nestedArray;
    for (let n=0; n < tensors.length; n++) {
      // console.log(this.heatmaps[n]);
      if (this.heatmaps[n].transpose) {
        nestedArray = nestedArrayFrom2dTensor(tensors[n].transpose());
      } else {
        nestedArray = nestedArrayFrom2dTensor(tensors[n]);
      }
      // console.log(nestedArray);
      updateMatrix(nestedArray, this.elementsConsumingTensors[n]);
    } 
  }

  // Returns position of bottom right corner
  addTensorHeatmap(tensor, tensorName, startingPos=[0,0], transpose=false) {
    // console.log('Visualizing tensor: ', tensorName, tensor);
    let nestedArray, pos, svgElem;
    if (transpose) {
      nestedArray = nestedArrayFrom2dTensor(tensor.transpose());
      pos = [startingPos[0] + tensor.shape[0] * this.squareSize_,
             startingPos[1] + tensor.shape[1] * this.squareSize_];
   } else {
      nestedArray = nestedArrayFrom2dTensor(tensor);
      pos = [startingPos[0] + tensor.shape[1] * this.squareSize_,
             startingPos[1] + tensor.shape[0] * this.squareSize_];
    }
    svgElem = createSvgElement(tensorName, this.g, startingPos[0], startingPos[1]); 
    visualizeMatrix(nestedArray, svgElem, this.squareSize_);
    this.elements.push(svgElem);
    this.elementsConsumingTensors.push(svgElem);
    let newHeatmap = new TensorHeatmap(tensorName, transpose);
    this.heatmaps.push(newHeatmap);

    // // Refresh current x and y positions 
    // if (stack) {
    //   this.yPos = pos[1];
    //   this.innerHeight = Math.max(this.innerHeight, this.yPos);
    // } else {
    //   this.xPos = pos[0];
    //   this.innerWidth = Math.max(this.innerWidth, this.xPos);
    //   this.innerHeight = Math.max(this.innerHeight, this.yPos, pos[1]);
    // }
    // this.xPos = pos[0];
    // this.yPos = pos[1];
    // console.log(tensorName, this.innerHeight);
    // console.log(tensorName, this.innerHeight);
    return pos;
  }

  // padX(numPaddingSquares) {
  //   this.xPos += numPaddingSquares * this.squareSize_;
  //   this.innerWidth = this.xPos;
  // }

  // padY(numPaddingSquares) {
  //   this.yPos += numPaddingSquares * this.squareSize_; 
  //   this.innerHeight = Math.max(this.innerHeight, this.yPos);
  // }

  addPlus(width=50, margin=10, startingPos=[0,0]) {
    const plusContainer = this.g.append('g')
        .attr('class', 'plus')
        .attr('width', width)
        .attr('height', width) 
        .attr('transform', 'translate(' + startingPos[0] + ',' +
              startingPos[1] + ')');

    plusContainer.append('line')
        .attr('class', 'plusLine')
        .attr('x1', margin)
        .attr('y1', width / 2)
        .attr('x2', width - margin)
        .attr('y2', width / 2);
    
    plusContainer.append('line')
        .attr('class', 'plusLine')
        .attr('x1', width / 2)
        .attr('y1', margin)
        .attr('x2', width / 2)
        .attr('y2', width - margin);
    
    this.xPos += width;
    this.innerWidth = this.xPos;
    return [startingPos[0] + width, startingPos[1] + width]
    //this.yPos += width;
  }


  addText(newText, startingPos=[0,0]) {
    this.g.append('text')
        .attr('class', 'tanh')
        .attr('x', startingPos[0])
        .attr('y', startingPos[1])
        .text(newText);
    this.xPos += 80;
    this.innerWidth = this.xPos;

    return [startingPos[0] + 60, startingPos[1] + 16]
  }


  addHadamard(size=8, startingPos=[0,0])  {
    const plusContainer = this.g.append('circle')
      .attr('class', 'circle')
      .attr('cx', startingPos[0] + size/2)
      .attr('cy', startingPos[1] + size/2 + 5) 
      .attr('r', size);

    this.xPos += size;
    this.innerWidth = this.xPos;
    return [startingPos[0] + size, startingPos[1] + 20]
    //this.yPos += size;
  }
}


// Accordion for Layers viz
function toggleShowRow() {
  // Toggle between adding and removing the 'active' class
  this.classList.toggle('active');
  // Show or hide the referenced row
  const row = document.getElementById(this.innerHTML);
  if (row.style.display === 'block') {
    row.style.display = 'none';
  } else {
    row.style.display = 'block';
  }
} 
  
