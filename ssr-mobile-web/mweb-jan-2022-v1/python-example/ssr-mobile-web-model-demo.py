#!/usr/bin/env python
# coding: utf-8

# # SSR Mobile Web Model Python Demo

# We'll see how to load the mweb-may-2020-v1 predictor and make predictions with it in Python. As a bonus, we also share a playground to play with the model and get a feel for its performance. The interactive UI demo only works in a notebook interface.
# 
# Simply run all the cells below to get started. 

get_ipython().system(' pip install -U -q tensorflow==2.2 jupyter-dash pandas')


import re
import logging
from pathlib import Path

import tensorflow as tf

from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

tf.__version__


logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(message)s')


class MWebMay2020Predictor:
    def __init__(self, modelDir):
        self.modelDir = modelDir;
        self.modelName = Path(modelDir).name
        self.model = None
        
        self._features = [
          'asn_number',
          'browser_major_version',
          'browser_major_version_na',
          'browser_name',
          'country_code',
          'dataCenter',
          'osFamily',
          'osMajor',
          'osMajor_na',
          'popId'
        ]


        # As saved in /jobs/nqsdev/lite/phase3/production_models/may-2020-model/best_model_nqs-lite-p30-cut-run2__HYPEROPT_AHB_60_SAMPLES_40_ITERATIONS/tfms-fixed.pickle for /jobs/nqsdev/lite/phase3/model_training_results/May2020-nqs-web-mapper-data/nqs-lite-p30-cut-run2__HYPEROPT_AHB_60_SAMPLES_40_ITERATIONS/ray_results/workspace/deepthought/nqs/src/linkedin/nqs/kong_model_training/tensorboard_data/ray_results/HYPEROPT_AHB_60_SAMPLES_40_ITERATIONS/DenseNQSModel_16_asn_hash_buckets=2201,batch_size=4096,cols_in_csv_file_order=['asn_number', 'country_code', 'popId', 'osFamily', _2020-06-30_00-25-161lyqlcwm/est_model_dir model
        self._defaults = {
          "browser_major_version": 75.0,
          "osMajor": 11.0,
          "asn_number": '**',
          "country_code": '**',
          "browser_name": '**',
          "osFamily": '**',
          "popId": '**',
          "dataCenter": '**'
        }

        self._normalizer = {
          "means": {"browser_major_version": 776.7030035555556, "osMajor": 10.416319973334147},
          "stds": {"browser_major_version": 120559.96265890554, "osMajor": 2.4117271070878434}
        }

    def loadModel(self):
        self.model = tf.saved_model.load(self.modelDir).signatures["predict"]
        
    def _normalizeNumericalFetaures(self, x):
        means = self._normalizer['means']
        stds = self._normalizer["stds"] 
        for feature in means:
            x[feature] = (float(x[feature]) - means[feature]) / stds[feature];
        return x;

    def _checkNA(self, value):
        res = value == None or value == '' or value == 'unknown'
        if isinstance(value, float) or isinstance(value, int):
            res = res or value < 0
        return res

    def _fillNA(self, x):
        for feature in x.keys():
            if self._checkNA(x[feature]):
                x[feature] = self._defaults[feature];
        return x;

    def _addNAFetaures(self, x):
        x["browser_major_version_na"] = 'False';
        x["osMajor_na"] = 'False';

        if self._checkNA(x["browser_major_version"]):
            x["browser_major_version_na"] = 'True';

        if self._checkNA(x["osMajor"]):
            x["osMajor_na"] = 'True';

        return x;
    
    def _convert_to_bytes(self, x):
        for feat, val in x.items():
            if isinstance(val, str):
                x[feat] = bytes(x[feat], 'utf-8')
        return x
    
    def prepareX(self, inp_example):
        model_input = tf.train.Example(features=tf.train.Features(feature={
            'country_code': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["country_code"]])),
            'popId': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["popId"]])),
            'osFamily': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["osFamily"]])),
            'browser_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["browser_name"]])),
            'dataCenter': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["dataCenter"]])),
            'browser_major_version_na': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["browser_major_version_na"]])),
            'osMajor_na': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["osMajor_na"]])),
            'asn_number': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["asn_number"]])),
            'browser_major_version': tf.train.Feature(float_list=tf.train.FloatList(value=[inp_example["browser_major_version"]])),
            'osMajor': tf.train.Feature(float_list=tf.train.FloatList(value=[inp_example["osMajor"]]))
        }))
        return model_input.SerializeToString()

    def preProcessInput(self, inp):
        x = {};
        for feature in self._features:
            x[feature] = inp.get(feature, None);

        x = self._addNAFetaures(x);
        x = self._fillNA(x);
        x = self._normalizeNumericalFetaures(x);
        x = self._convert_to_bytes(x)
        return x;
    
    
    def predict(self, rawInput):
        """
        * Process the input and make predictions on it
        * @param {object} rawInput {[name: string]: tf.Tensor} dictionary
        * @returns {class1: probability1, class2: probability2, ...}
        """
        if not self.model:
            self.loadModel()

        inp = self.preProcessInput(rawInput);
        logging.debug(f"Model input: {inp}")
        x = self.prepareX(inp)
        logging.debug(f"Model (x): {x}")
        output = self.model(examples=tf.constant([x]))
        return output


def make_prediction(predictor, inp):
    p = predictor.predict(inp)
    scores = p['probabilities'].numpy()[0]
    return {i: score for i, score in enumerate(scores)} # return the probability for each class


# the below model is the TF Python equivalent of JS' saved model
MODEL_PATH = "../models/py-saved-model/"


predictor = MWebMay2020Predictor(MODEL_PATH)
predictor.loadModel()


# Make some predictions

make_prediction(predictor, 
    {
        'asn_number': '40793',  'browser_major_version': '67',  'browser_name': 'chrome',  
         'country_code': 'us',  'dataCenter': '**',  'osFamily': 'Android',  'osMajor': '6',  'popId': '**'
    }
)


# Some example inputs to try, while getting started,
# ```json
# {'asn_number': '40793',  'browser_major_version': '67',  'browser_name': 'chrome',  'country_code': 'us',  'dataCenter': '**',  'osFamily': 'Android',  'osMajor': '6',  'popId': '**'}
# {'asn_number': '3352',  'browser_major_version': '13',  'browser_name': 'safari',  'country_code': 'es',  'dataCenter': '**',  'osFamily': 'iOS',  'osMajor': '13',  'popId': '**'}
# {'asn_number': '40793',  'browser_major_version': '67',  'browser_name': 'chrome',  'country_code': 'us',  'dataCenter': '**',  'osFamily': 'Android',  'osMajor': '6',  'popId': '**'}
# ```

# ## Interactive UI

def design_inline_form_control(label:str, input_type:str="text", default_val="", readonly=False):
    input_id = re.sub(r"\s+", "", label)
    div = html.Div([
        html.Div([
            html.Label(label, className='col-form-label', htmlFor=input_id)
        ], className="col-md-3"),
        html.Div([            
            dcc.Input(id=input_id, value=default_val, type=input_type, required=True, 
                      className="form-control", readOnly=readonly)
        ], className="col-auto")
    ], className="row g-3 mb-3 align-items-center")
    return div, input_id


app = JupyterDash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"])

asn_div, a_id = design_inline_form_control("ASN number", "number", 3352)
browser_version_div, bv_id = design_inline_form_control("Browser major version", "number", 14)
browser_name_div, bn_id = design_inline_form_control("Browser name", default_val='safari')
country_div, cc_id = design_inline_form_control("Country code", default_val='ca')
os_family_div, os_id = design_inline_form_control("OS Family", default_val='iOS')
os_major_div, osm_id = design_inline_form_control("OS Major version", "number", 14)
dc_div, _ = design_inline_form_control("Datacenter", default_val="**", readonly=True)
pop_div, _ = design_inline_form_control("POP", default_val="**", readonly=True)
        
app.layout = html.Div([
    html.H1("Performance Quality Predictor", className="mb-5"),
    html.P("The model is live and ready! Try changing any of the values below and see the prediction at the end.", className="text-muted"),
    html.Div([
        asn_div, browser_version_div, browser_name_div, country_div, os_family_div, os_major_div, dc_div, pop_div
    ]),
    html.P([
        "The model thinks the performance quality for the above request to be, ",
        html.Mark("Good", id="result_class"),
        " with ",
        html.Mark("85%", id="result_prob"),
        " confidence."
    ], className="lead mt-4")
])

@app.callback(
    [Output("result_class", 'children'), Output("result_prob", 'children')],
    [Input(a_id, "value"), Input(bv_id, "value"), 
     Input(bn_id, "value"), Input(cc_id, "value"), 
     Input(os_id, "value"), Input(osm_id, "value")]
)
def update_figure(asn_number:int, browser_version:int, browser_name:str, country_code:str, os_family:str, os_major:int):
    inp = {
        'asn_number': f"{asn_number}",
        'browser_major_version': browser_version,
        'browser_name': browser_name,
        'country_code': country_code,
        'dataCenter': '**',
        'osFamily': os_family,
        'osMajor': os_major,
        'popId': '**'
    }
    pred = make_prediction(predictor, inp)
    good_prob = pred[0]
    bad_prob = pred[1] # or 1 - good_prob
    if good_prob > bad_prob:
        return "Good", f"{good_prob:.2%}"
    else:
        return "Bad", f"{bad_prob:.2%}"

# Run app and display result inline in the notebook
app.run_server(mode='inline', height=630)



