const express = require('express');
const app = express();
const { MWebJan2022Predictor } = require('./predictors');

async function main() {
  // load the model so it's ready
  const predictor = new MWebJan2022Predictor(__dirname + '/../models/nodejs-saved-model');
  await predictor.loadModel();

  // Make all the files in 'www' available.
  app.use(express.static('www'));

  app.get('/api/performanceQuality', (req, res) => {
    const {
      asn,
      country_code,
      browser_name,
      browser_major_version,
      osFamily,
      osMajor
    } = req.query;
    const result = predictor.predict({asn, country_code, browser_name, browser_major_version, osFamily, osMajor});
    res.setHeader('Content-Type', 'application/json');
    res.send(result);
  });

  app.get('/', (req, res) => {
    res.sendFile(__dirname + '/www/index.html');
  });

  // Listen for requests.
  const listener = app.listen(3001, () => {
    console.log('Your app is listening on port ' + listener.address().port);
  });
}

main();

