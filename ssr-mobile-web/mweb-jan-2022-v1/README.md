# Mobile Web Jan 2022 V1

## About the Model

### Model Input

| Model Input | Description | Some Examples |
| ----------- | ----------- | ------------- |
| Country code | ISO 3166-1 alpha-2 country code ([more details](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)) | us, in, br |
| OS Family | Name of the operating system | iOS, Android, Windows |
| OS version | Major version number of the OS | 14, 8 |
| Browser | Name of browser | Chrome, Safari |
| Browser version | Major version number of the browser | 14, 74 |
| ASN number | [Autonomous System Number](https://en.wikipedia.org/wiki/Autonomous_system_(Internet)), like ISP. Check the code example on how to obtain this from IP address | “7922” |

OS and browser info is extracted from the user agent using the [UA Parser library](https://mvnrepository.com/artifact/ua_parser/ua-parser/1.3.0). Using any other way to extract this information may work but it may lead to [train-serve data skew](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift).

### Model Output

The model is trained to return the page load time class. For now, we have two classes:

1. Less than 1200ms and
2. Greater than or equal to 1200ms, 

but we may expand them to more classes in the future based on the use cases. As shown in the predictor-runner.ipynb example, the [TF predictor](https://github.com/tensorflow/tensorflow/blob/63f17d0fe1192eff0aa47faae5d15ec7aa02490a/tensorflow/python/saved_model/load.py#L850) returns a probability distribution of these PLT classes. We could simply pick the class with the highest probability as the model’s prediction. SavedModels from Estimators section from this [guide](https://github.com/tensorflow/docs/blob/e9f1ce05852b13e9335860d93aa28f0782b60ddc/site/en/guide/estimator.ipynb) is another end to end example of this pattern.

## How to use it

We currently have examples of how to use this model in Python and Node.js:

- Python: [README.md](python-example/README.md)
- Node.js: [README.md](nodejs-example/README.md)

## FAQs

**How was the model built?**

A deep neural network model is trained on historical [RUM](https://developer.mozilla.org/en-US/docs/Web/Performance/Rum-vs-Synthetic#Real_User_Monitoring) data of LinkedIn Lite. Lite is a [server side rendered](https://engineering.linkedin.com/blog/2018/03/linkedin-lite--a-lightweight-mobile-web-experience) application whose [onLoad](https://developer.mozilla.org/en-US/docs/Web/API/GlobalEventHandlers/onload) event is used as a proxy for page load time (PLT). Standard features like browser, OS, ASN etc. as inputs and bucketed PLT as target are fed to a [tf.estimator.DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) for [training](https://developers.google.com/machine-learning/glossary/#training). The trained model is exported to disk in [saved model](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk) format, which we are distributing as part of this repo. We are working on a detailed blog on how we trained 100s of models in an automated manner to find the best one on our [Engineering Blog](https://engineering.linkedin.com/blog). In the meantime, you can checkout this [presentation](http://bit.ly/ray-at-linkedin) and [video](https://youtu.be/0Z0Th9ySIfs?t=761) which contains much of the same content.

**How was the model trained? How much training data was used? How far back in time did it go?**

Around 1M randomly sampled LinkedIn Lite’s RUM data from the month of May 2020 was used for training.

