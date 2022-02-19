# Mobile Web Jan 2022 V1

## About the Model

The model was built and trained from around 25M random samples of LinkedIn Lite’s [RUM](https://developer.mozilla.org/en-US/docs/Web/Performance/Rum-vs-Synthetic#Real_User_Monitoring) data from the month of January 2022. Lite is LinkedIn's main mobile web server application.

### Model Input

| Model Input | Description | Some Examples |
| ----------- | ----------- | ------------- |
| Country code | ISO 3166-1 alpha-2 country code ([more details](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)) | us, in, br |
| OS Family | Name of the operating system | iOS, Android, Windows |
| OS version | Major version number of the OS | 14, 8 |
| Browser | Name of browser | Chrome, Safari |
| Browser version | Major version number of the browser | 14, 74 |
| ASN number | [Autonomous System Number](https://en.wikipedia.org/wiki/Autonomous_system_(Internet)), like your Internet Service Provider | “7922” |

OS and browser info is extracted from the user agent using the [UA Parser library](https://mvnrepository.com/artifact/com.github.ua-parser/uap-java/1.4.3). Using any other way to extract this information may work but it may lead to [train-serve data skew](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift).

Using [Digital Element](https://www.digitalelement.com/resources/faq/)'s Jan 2022 database, we extracted ASN numbers from IP Addresses. You may have to purchase their license to be absolutely match to what we did. However, any such accurate translation service should work. While trying out our demos, use any free website like [this one](https://hackertarget.com/as-ip-lookup/) or [this one](https://mxtoolbox.com/asn.aspx) to obtain the ASN number for the IP Address of your interest. 

### Model Output

The model is trained to return the page load time class. For now, we have two classes:

1. Less than 1300ms and
2. Greater than or equal to 1300ms, 

but we may expand them to more classes in the future based on the use cases. As shown in the [ssr-mobile-web-modile-demo.pynb](python-example/ssr_mobile_web_model_demo.ipynb) example, the [TF predictor](https://github.com/tensorflow/tensorflow/blob/63f17d0fe1192eff0aa47faae5d15ec7aa02490a/tensorflow/python/saved_model/load.py#L850) returns a probability distribution of these PLT classes. We could simply pick the class with the highest probability as the model’s prediction. SavedModels from Estimators section from this [guide](https://github.com/tensorflow/docs/blob/e9f1ce05852b13e9335860d93aa28f0782b60ddc/site/en/guide/estimator.ipynb) is another end to end example of this pattern.

## How to use it

We currently have examples of how to use this model in Python and Node.js:

- Python: [README.md](python-example/README.md)
- Node.js: [README.md](nodejs-example/README.md)

## FAQs

**How was the model built?**

A deep neural network model is trained on historical [RUM](https://developer.mozilla.org/en-US/docs/Web/Performance/Rum-vs-Synthetic#Real_User_Monitoring) data of LinkedIn Lite. Lite is a [server side rendered](https://engineering.linkedin.com/blog/2018/03/linkedin-lite--a-lightweight-mobile-web-experience) application whose [onLoad](https://developer.mozilla.org/en-US/docs/Web/API/GlobalEventHandlers/onload) event is used as a proxy for page load time (PLT). Standard features like browser, OS, ASN etc. as inputs and bucketed PLT as target are fed to a [tf.estimator.DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) for [training](https://developers.google.com/machine-learning/glossary/#training). The trained model is exported to disk in [saved model](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk) format, which we are distributing as part of this repo. We are working on a detailed blog on how we trained 100s of models in an automated manner to find the best one on our [Engineering Blog](https://engineering.linkedin.com/blog). In the meantime, you can checkout this [presentation](http://bit.ly/ray-at-linkedin) and [video](https://youtu.be/0Z0Th9ySIfs?t=761) which contains much of the same content.
