var weights = tf.randomNormal([3, 10]);
var epochs = 250;

const oneHot = (val, categoryCount) =>
  Array.from(tf.oneHot(val, categoryCount).dataSync());

var raw_inputs = [];
var raw_outputs = [];
// prettier-ignore
// let raw_inputs = [0.58457386, 0.873974, 0.2523505, 0.7349865, 0.11772124, 0.1399177, 0.091900155, 0.96005094, 0.30265996, 0.43436098, 0.2061948, 0.84002817, 0.6674902, 0.2572348, 0.4988692, 0.8522466, 0.044111863, 0.11403932, 0.4911531, 0.2768156, 0.7581087, 0.5256973, 0.9096914, 0.64478236, 0.013603632, 0.47027507, 0.71730167, 0.14791192, 0.27054757, 1.0, 0.14791192, 0.27054757, 0.43677083, 0.14791192, 0.90336007, 0.43677083, 0.14791192, 0.90336007, 0.08145834, 1.0, 0.90336007, 0.08145834, 1.0, 0.90336007, 0.41609374, 1.0, 1.0, 0.61364585, 1.0, 1.0, 0.0, 1.0, 1.0, 0.19864583, 1.0, 0.8985937, 0.19864583, 0.50216556, 0.020547176, 0.4865946, 0.50216556, 0.020547176, 0.0, 1.0, 0.020547176, 0.0, 0.14614584, 0.020547176, 0.0, 0.45239583, 0.020547176, 0.0, 0.90593755, 0.2203125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034635417, 0.0575, 1.0, 1.0, 1.0, 1.0, 0.93609375, 1.0, 0.94463545, 0.93609375, 1.0, 0.94463545, 0.93609375, 0.9115104, 0.94463545, 0.8410417, 0.9115104, 0.575625, 0.43520835, 0.04921875, 0.5276844, 0.35780025, 0.21673357, 0.5276844, 0.40316483, 0.21673357, 0.5911219, 0.40316483, 0.21673357, 0.5911219, 0.40316483, 0.29829606, 0.5911219, 0.40316483, 0.15142109, 0.5911219, 0.40316483, 0.0, 0.7576323, 0.37868565, 0.0, 0.5296115, 0.37868565, 0.0]
// prettier-ignore
// let raw_outputs = [2, 0, 2, 4, 6, 0, 4, 2, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 0, 0, 0, 0, 0, 7, 7, 8, 8, 8, 8, 8, 3, 5, 5, 5, 5, 5, 5, 5, 5]

function lossFunction(yhat, outputs) {
  return tf.losses.softmaxCrossEntropy(outputs, yhat);
}

function lossPrime(yhat, outputs) {
  return outputs.sub(yhat);
}

function predict(inputs) {
  // weights.print();
  return tf.softmax(tf.matMul(inputs, weights));
}

function train() {
  let y = raw_outputs.map((y) => oneHot(y, 10));

  let inputs = tf.tensor(raw_inputs, [raw_inputs.length / 3, 3]);
  let outputs = tf.tensor(y, [y.length, 10]);

  for (let i = 1; i <= epochs; i++) {
    let yhat = predict(inputs);
    let loss = lossFunction(yhat, outputs);

    let adj = loss.mul(lossPrime(yhat, outputs));
    let adjustments = inputs.transpose().dot(adj);

    weights = weights.add(adjustments);
  }
  document.getElementById("outputLabel").innerHTML = "Done!";
}

function loadPretrained() {
  // prettier-ignore
  weights = tf.tensor ([[92.5186615, -87.8690033, -155.7975769, 94.7636948 , 8.9406767   , 6.9089098  , 35.9979286  , 66.7746429 , -37.9027901, -28.0783501],
    [-85.557724, 157.7426147, 37.9013023  , -10.3656979, 69.6718063  , -59.1021347, -127.0380249, -9.6802902 , -37.5198479, 62.6954079 ],
    [-31.682806, -67.1004791, 117.0795059 , -74.8525391, -106.7242737, 104.1632156, 10.2801323  , -57.9854507, 62.9147224 , 41.2350159 ]])
  document.getElementById("outputLabel").innerHTML =
    "Model Loaded. Start Classifying!";
}

function sliderChanged(val) {
  let r = parseFloat(document.getElementById("red").value);
  let g = parseFloat(document.getElementById("green").value);
  let b = parseFloat(document.getElementById("blue").value);
  var bgColor = "rgb(" + r + "," + g + "," + b + ")";

  document.getElementById("colorView").style.background = bgColor;
}

function randomColor() {
  var x = Math.floor(Math.random() * 255);
  var y = Math.floor(Math.random() * 255);
  var z = Math.floor(Math.random() * 255);
  var bgColor = "rgb(" + x + "," + y + "," + z + ")";

  document.getElementById("colorView").style.background = bgColor;
  document.getElementById("red").value = x;
  document.getElementById("green").value = y;
  document.getElementById("blue").value = z;
}

var vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
// prettier-ignore
var labels = ["Red","Green","Blue","Orange","Yellow","Purple","Pink","Brown","Black","White",];

function save(index) {
  let red = parseFloat(document.getElementById("red").value) / 255.0;
  let green = parseFloat(document.getElementById("green").value) / 255.0;
  let blue = parseFloat(document.getElementById("blue").value) / 255.0;
  vals[index] += 1;
  raw_inputs.push(red);
  raw_inputs.push(green);
  raw_inputs.push(blue);
  raw_outputs.push(index);

  document.getElementById(index + "-count").innerHTML = vals[index];

  let count = vals.reduce((a, b) => a + b, 0);
  document.getElementById("totalTrainingData").innerHTML =
    "Total Training Data: " + count;
}

function classify() {
  let red = parseFloat(document.getElementById("red").value) / 255.0;
  let green = parseFloat(document.getElementById("green").value) / 255.0;
  let blue = parseFloat(document.getElementById("blue").value) / 255.0;

  let prediction = predict(tf.tensor2d([[red, green, blue]]));

  let vals = prediction.dataSync();
  var selectedIndex = 0;
  var largestVal = 0;

  for (let i = 0; i < vals.length; i++) {
    if (vals[i] > largestVal) {
      largestVal = vals[i];
      selectedIndex = i;
    }
  }

  document.getElementById("outputLabel").innerHTML = labels[selectedIndex];
}
