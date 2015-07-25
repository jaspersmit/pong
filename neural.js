function logistic(x) {
    return 1 / (1 + Math.exp(-x));
}

var Neural = function(numInputNodes, numHiddenNodes, numOutputNodes) {
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
    this.outputValues = new Array(numOutputNodes);
}


Neural.prototype.eval = function(inputs) {
    var i;
    var j;
    for(i = 0; i < this.numHiddenNodes; i++) {
        this.hiddenValues[i] = 0;
    }
    for(i = 0; i < this.numInputNodes; i++) {
        for(j = 0; j < this.numHiddenNodes; j++) {
            var w = this.layer1Weights[i * this.numHiddenNodes + j]
            var input = inputs[i];
            this.hiddenValues[j] += w * input;
        }
    }
    for(i = 0; i < this.numHiddenNodes; i++) {
        this.hiddenValues[i] = logistic(this.hiddenValues[i]);
    }
    for(i = 0; i < this.numOutputNodes; i++) {
        this.outputValues[i] = 0;
    }
    for(i = 0; i < this.numHiddenNodes; i++) {
        for(j = 0; j < this.numOutputNodes; j++) {
            var w = this.layer2Weights[i * this.numOutputNodes + j]
            var hiddenValue = this.hiddenValues[i];
            this.outputValues[j] += w * hiddenValue;
        }
    }
    for(i = 0; i < this.numOutputNodes; i++) {
        this.outputValues[i] = logistic(this.outputValues[i]);
    }
    return this.outputValues;
}
    
