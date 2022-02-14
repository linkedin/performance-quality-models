const path = require("path");
const tf = require("@tensorflow/tfjs-node");

class PredictorBase {
  constructor(modelDir) {
    this.modelDir = modelDir;
    this.modelName = path.basename(modelDir);
    this.model = null;
  }

  async loadModel() {
    this.model = await tf.node.loadSavedModel(this.modelDir);
  }

  preProcessInput(input) {
    throw new Error("Not Implemented");
  }

  /**
   * Wraps the feature values as tensor arrays
   * Types are inferred by tfjs
   * Keys to input should match with what the model expects
   * @param {object} input final input dictionary ready to be passed to the model
   * @returns a dictionary of tensor arrays as required by the model
   */
  prepareX(input) {
    const x = {};
    for (const feature of Object.keys(input)) {
      const value = input[feature];
      x[feature] = tf.tensor([value], [1, 1]); // explicitly ensure it is not a rank 0 tensor
    }
    return x;
  }

  /**
   * Process the input and make predictions on it
   * @param {object} rawInput {[name: string]: tf.Tensor} dictionary
   * @returns {class1: probability1, class2: probability2, ...}
   */
  predict(rawInput) {
    if (!this.model) {
      throw new Error(
        `Model '${this.modelName}' is not loaded. Please load it first.`
      );
    }
    const result = tf.tidy(() => {
      const input = this.preProcessInput(rawInput);
      const x = this.prepareX(input);
      const output = this.model.predict(x, {});
      const probs = Array.from(output.probabilities.dataSync());
      const classes = Array.from(output.all_class_ids.dataSync());
      const result = Object.fromEntries(
        classes.map((classId, i) => [classId, probs[i]])
      );
      return result;
    });

    for (const [key, value] of Object.entries(result)) {
      if (typeof value !== "number" || isNaN(value)) {
        throw new Error(
          `Invalid confidence score (${value}) causing bad prediction result: ${JSON.stringify(
            result
          )}`
        );
      }
    }

    return result;
  }
}

class MWebJan2022Predictor extends PredictorBase {
  constructor(modelDir) {
    super(modelDir);
    this._features = [
      "asn_number",
      "browser_major_version",
      "browser_major_version_na",
      "browser_name",
      "country_code",
      "osfamily",
      "osmajor",
      "osmajor_na"
    ];
    this._defaults = {
      "browser_major_version": 15.0,
      "osmajor": 14.0,
      "asn_number": '**',
      "country_code": '**',
      "browser_name": '**',
      "osfamily": '**'
    };
    this._normalizer = {
      "means": { "browser_major_version": 52.65782220933843, "osmajor": 13.372263709715911 },
      "stds": { "browser_major_version": 41.48294747389074, "osmajor": 2.376855002582524 }
    };
  }

  async loadModel() {
    this.model = await tf.node.loadSavedModel(
      this.modelDir,
      ["serve"],
      "predict"
    );
  }

  _normalizeNumericalFetaures(x) {
    const { means, stds } = this._normalizer;
    for (const feature in means) {
      x[feature] = (parseFloat(x[feature]) - means[feature]) / stds[feature];
    }
    return x;
  }

  _checkNA(value) {
    return (
      value === null ||
      value === undefined ||
      value < 0 ||
      value === "" ||
      value === "unknown"
    );
  }

  _fillNA(x) {
    for (const feature of Object.keys(x)) {
      if (this._checkNA(x[feature])) {
        x[feature] = this._defaults[feature];
      }
    }
    return x;
  }

  _addNAFetaures(x) {
    x.browser_major_version_na = "False";
    x.osmajor_na = "False";

    if (this._checkNA(x.browser_major_version)) {
      x.browser_major_version_na = "True";
    }

    if (this._checkNA(x.osmajor)) {
      x.osmajor_na = "True";
    }
    return x;
  }

  preProcessInput(input) {
    let x = {};
    for (const feature of this._features) {
      x[feature] = input[feature];
    }
    x = this._addNAFetaures(x);
    x = this._fillNA(x);
    x = this._normalizeNumericalFetaures(x);
    return x;
  }
}

module.exports = { MWebJan2022Predictor };
