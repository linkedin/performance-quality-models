#!/usr/bin/env python
# coding: utf-8

# # SSR Mobile Web Model Python Demo

# We'll see how to load the mweb-jan-2022-v1 predictor and make predictions with it in Python. As a bonus, we also share a playground to play with the model and get a feel for its performance. The interactive UI demo only works in a notebook interface.
# 
# Simply run all the cells below to get started. 

# In[1]:


get_ipython().system(' pip install -U -q pip && pip install -U -q tensorflow==2.5 pandas')


# In[2]:


import re
import logging
from pathlib import Path

import tensorflow as tf

tf.__version__


# In[3]:


logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')


# In[11]:


# the below model is the TF Python equivalent of JS' saved model
MODEL_PATH = "../models/py-saved-model"


# Setup the notebook for Google Colab. This cell can be ignored if not on colab.google.com

# In[6]:


try:
    import google.colab
    import subprocess
    
    clone_cmd_res = subprocess.run(
      "git clone -l -s https://github.com/linkedin/performance-quality-models.git performance-quality-models",
      shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    
    if clone_cmd_res.returncode != 0:
        raise Exception(clone_cmd_res.stderr)
    
    get_ipython().run_line_magic('cd', 'performance-quality-models')
    
    MODEL_PATH = "./ssr-mobile-web/mweb-jan-2022-v1/models/py-saved-model"
except:
    logging.warning("Ignore this warning if not on colab.google.com", exc_info=True)


# Define a Predictor class which loads the model and transforms the data into a form that model can understand.

# In[7]:


class MWebJan2022Predictor:
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
          'osfamily',
          'osmajor',
          'osmajor_na'
        ]

        self._defaults = {
            "browser_major_version": 15.0, 
            "osmajor": 14.0,
            "asn_number": '**',
            "country_code": '**',
            "browser_name": '**',
            "osfamily": '**',
        }

        self._normalizer = {
            "means": {"browser_major_version": 52.65782220933843, "osmajor": 13.372263709715911}, 
            "stds": {"browser_major_version": 41.48294747389074, "osmajor": 2.376855002582524}
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
        x["osmajor_na"] = 'False';

        if self._checkNA(x["browser_major_version"]):
            x["browser_major_version_na"] = 'True';

        if self._checkNA(x["osmajor"]):
            x["osmajor_na"] = 'True';

        return x;
    
    def _convert_to_bytes(self, x):
        for feat, val in x.items():
            if isinstance(val, str):
                x[feat] = bytes(x[feat], 'utf-8')
        return x
    
    def prepareX(self, inp_example):
        model_input = tf.train.Example(features=tf.train.Features(feature={
            'country_code': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["country_code"]])),
            'osfamily': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["osfamily"]])),
            'browser_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["browser_name"]])),
            'browser_major_version_na': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["browser_major_version_na"]])),
            'osmajor_na': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["osmajor_na"]])),
            'asn_number': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp_example["asn_number"]])),
            'browser_major_version': tf.train.Feature(float_list=tf.train.FloatList(value=[inp_example["browser_major_version"]])),
            'osmajor': tf.train.Feature(float_list=tf.train.FloatList(value=[inp_example["osmajor"]]))
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


# In[8]:


def make_prediction(predictor, inp):
    p = predictor.predict(inp)
    scores = p['probabilities'].numpy()[0]
    return {i: score for i, score in enumerate(scores)} # return the probability for each class


# In[12]:


predictor = MWebJan2022Predictor(MODEL_PATH)
predictor.loadModel()


# Make some predictions

# In[13]:


make_prediction(predictor, 
    {
        'asn_number': '40793',  'browser_major_version': '67',  'browser_name': 'chrome',  
         'country_code': 'us',  'osfamily': 'Android',  'osmajor': '6'
    }
)


# A result, `{0: 0.0106238695, 1: 0.9893762}` implies that the model is 98.94% sure that the given is input configuration of the device and network will have **poor** performance quality (i.e page load time > 950ms). In this case we disable all aggresive optimizations. 
# 
# To read it the other way, the model is 1.06% sure (LOL) that the input configuration has a **good** performance, i.e. page load time <= 950ms.

# Some example inputs to try, while getting started,
# ```json
# {'asn_number': '40793',  'browser_major_version': '67',  'browser_name': 'chrome',  'country_code': 'us',  
#     'osfamily': 'Android',  'osmajor': '6'}
# {'asn_number': '3352',  'browser_major_version': '13',  'browser_name': 'safari',  'country_code': 'es',  
#     'osfamily': 'iOS',  'osmajor': '13'}
# {'asn_number': '40793',  'browser_major_version': '67',  'browser_name': 'chrome',  'country_code': 'us',  
#     'osfamily': 'Android',  'osmajor': '6'}
# ```

# ## Interactive UI
# 
# To understand the model's behavior a bit more, use the below interactive UI. The model predicts on every keystroke. We can afford to do it, because it is so fast!

# In[14]:


get_ipython().system(' pip install -U -q pip && pip install -U -q dash==2.0.0 jupyter-dash==0.4.0')


# In[15]:


from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


# In[16]:


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


# In[17]:


app = JupyterDash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"])

asn_div, a_id = design_inline_form_control("ASN number", "number", 3352)
browser_version_div, bv_id = design_inline_form_control("Browser major version", "number", 14)
browser_name_div, bn_id = design_inline_form_control("Browser name", default_val='safari')
country_div, cc_id = design_inline_form_control("Country code", default_val='ca')
os_family_div, os_id = design_inline_form_control("OS Family", default_val='iOS')
os_major_div, osm_id = design_inline_form_control("OS Major version", "number", 14)
        
app.layout = html.Div([
    html.H1("Performance Quality Predictor", className="mb-5"),
    html.P("The model is live and ready! Try changing any of the values below and see the prediction at the end.", className="text-muted"),
    html.Div([
        asn_div, browser_version_div, browser_name_div, country_div, os_family_div, os_major_div
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
        'osfamily': os_family,
        'osmajor': os_major,
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


# In[ ]:




