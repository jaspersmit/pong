function logistic(x) {
    return 1 / (1 + Math.exp(-x));
}

function logisticDerivative(x) {
    return logistic(x) * (1 - logistic(x));
}   

var Neural = function(numInputNodes, numHiddenNodes, numOutputNodes) {
    this.errors = new Array(800);
    this.errorPointer = 0;
    this.numInputNodes = numInputNodes;
    this.numHiddenNodes = numHiddenNodes;
    this.numOutputNodes = numOutputNodes;
    this.layer1Weights = new Array(numInputNodes * numHiddenNodes);
    var i;
    for(i = 0; i < numInputNodes * numHiddenNodes; i++) {
        this.layer1Weights[i] = 2 * Math.random() - 1;
    }
    this.layer2Weights = new Array(numHiddenNodes * numOutputNodes);
    for(i = 0; i < numHiddenNodes * numOutputNodes; i++) {
        this.layer2Weights[i] = 2 * Math.random() - 1;
    }
    this.hiddenValues = new Array(numHiddenNodes);
    this.hiddenValuesNet = new Array(numHiddenNodes);
    this.outputValues = new Array(numOutputNodes);
    this.outputValuesNet = new Array(numOutputNodes);
}


Neural.prototype.eval = function(inputs) {
    var i;
    var j;
    for(i = 0; i < this.numHiddenNodes; i++) {
        this.hiddenValuesNet[i] = 0;
    }
    for(i = 0; i < this.numInputNodes; i++) {
        for(j = 0; j < this.numHiddenNodes; j++) {
            var w = this.layer1Weights[i * this.numHiddenNodes + j]
            var input = inputs[i];
            this.hiddenValuesNet[j] += w * input;
        }
    }
    for(i = 0; i < this.numHiddenNodes; i++) {
        this.hiddenValues[i] = logistic(this.hiddenValuesNet[i]);
    }
    for(i = 0; i < this.numOutputNodes; i++) {
        this.outputValuesNet[i] = 0;
    }
    for(i = 0; i < this.numHiddenNodes; i++) {
        for(j = 0; j < this.numOutputNodes; j++) {
            var w = this.layer2Weights[i * this.numOutputNodes + j]
            var hiddenValue = this.hiddenValues[i];
            this.outputValuesNet[j] += w * hiddenValue;
        }
    }
    for(i = 0; i < this.numOutputNodes; i++) {
        this.outputValues[i] = logistic(this.outputValuesNet[i]);
    }
    return this.outputValues;
}

Neural.prototype.train = function(inputs, expectedOutput, learningRate) {
    //Get gradients layer 2
    var i;
    var j;
    var k;
    this.eval(inputs);
    var totalError = 0;
    for(i = 0; i < this.numHiddenNodes; i++) {
        for(j = 0; j < this.numOutputNodes; j++) {
            var inputValue = this.hiddenValues[i]; 
            var netto = this.outputValuesNet[i];
            
            var gradient = 
                (this.outputValues[j] - expectedOutput[j])
                * logisticDerivative(netto) 
                * inputValue;

            this.layer2Weights[i * this.numHiddenNodes + j] -= learningRate * gradient;
            
        }
    }

    //Learn layer 2
    for(i = 0; i < this.numInputNodes; i++) {
        for(j = 0; j < this.numHiddenNodes; j++) {
            var inputValue = inputs[i]; 
            var netto = this.hiddenValuesNet[j];

            var gradient = 0;
            for(k = 0; k < this.numOutputNodes; k++) {
                var outputNetto = this.outputValuesNet[k];
                
                var weight = this.layer2Weights[k * this.numHiddenNodes + j];

                gradient += 
                    (this.outputValues[k] - expectedOutput[k])
                    * logisticDerivative(outputNetto)
                    * weight
                    * logisticDerivative(netto)
                    * inputValue;
            }
            this.layer1Weights[i * this.numHiddenNodes + j] -= learningRate * gradient;
        }
    }
    
    for(j = 0; j < this.numOutputNodes; j++) {
        totalError += (this.outputValues[j] - expectedOutput[j]) * (this.outputValues[j] - expectedOutput[j]);
    }

    this.errors[this.errorPointer] = totalError;
    this.errorPointer = (this.errorPointer + 1) % 800;

}
    
