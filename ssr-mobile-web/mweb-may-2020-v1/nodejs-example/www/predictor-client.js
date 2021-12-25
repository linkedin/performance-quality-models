(function() {

  function getFeatures() {
    const asn_number = $('#asn-input').val().trim();
    const country_code = $('#country-code-input').val().trim();
    const browser_name = $('#browser-name-input').val().trim();
    const browser_major_version = $('#browser-major-version-input').val().trim();
    const osFamily = $('#os-family-input').val().trim();
    const osMajor = $('#os-major-version-input').val().trim();

    const features = { asn_number, country_code, browser_name, browser_major_version, osFamily, osMajor };
    for (const k in features) {
      if (features[k] === '') {
        delete features[k];
      }
    }
    return features;
  }

  $('#api-form').submit(event => {
    $('#api-url').text('');
    $('#api-result').text('');

    const features = getFeatures();
    const queryString = new URLSearchParams(features);
    const apiUrl = `${window.location.origin}/api/performanceQuality?${queryString}`;
    $('#api-url').text(apiUrl);

    $.getJSON(apiUrl, data => {
      $('#api-result').text(JSON.stringify(data));
    });

    event.preventDefault();
  });

})();
