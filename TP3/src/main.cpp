
#include "DHT.h"
#include <Arduino.h>

#define DHTPIN 2      // Digital pin connected to the DHT sensor
#define DHTTYPE DHT22 // DHT 22  (AM2302), AM2321
DHT dht(DHTPIN, DHTTYPE);

const int N_FEATURES = 12;
const float MEAN[N_FEATURES] = {25.3, 60.4, 0.1, 420.0, 12000.0, 19000.0, 940.0, 0.0, 0.0, 0.0, 0.0, 0.0};
const float STD[N_FEATURES] = {3.5, 15.2, 0.5, 100.0, 2000.0, 2500.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0};
const float WEIGHTS[N_FEATURES] = {1.12, 0.78, 0.02, 0.15, -0.003, 0.001, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0};
const float BIAS = -0.82; 

float X[N_FEATURES] = {20.0, 57.36, 0, 400, 12306, 18520, 939.735, 0.0, 0.0, 0.0, 0.0, 0.0}; // Input features

float standardize(float x_raw, int idx) {
  return (x_raw - MEAN[idx]) / STD[idx];
}

float sigmoid(float z) {
  return 1.0 / (1.0 + exp(-z));
}

float predict(float features[]) {
  float z = 0.0;
  for (int i = 0; i < N_FEATURES; i++) {
    z += WEIGHTS[i] * features[i];
  }
  z += BIAS;
  return sigmoid(z);
}
void setup()
{
  Serial.begin(9600);
  Serial.println(F("DHTxx test!"));
  dht.begin();
}

void loop()
{
  delay(2000);

  // Reading temperature or humidity takes about 250 milliseconds!
  // Sensor readings may also be up to 2 seconds 'old' (its a very slow sensor)
  float h = dht.readHumidity();
  // Read temperature as Celsius (the default)
  float t = dht.readTemperature();
  // Read temperature as Fahrenheit (isFahrenheit = true)
  float f = dht.readTemperature(true);

  // add data to input array
  X[0] = t;
  X[1] = h;

  // Check if any reads failed and exit early (to try again).
  if (isnan(h) || isnan(t) || isnan(f))
  {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  // TODO: Add code to standardize the inputs
  float X_scaled[N_FEATURES];
    for (int i = 0; i < N_FEATURES; i++) {
      X_scaled[i] = standardize(X[i], i);
    }
  // TODO: Add code to compute the output of wx + b

  // TODO: Add code to apply the sigmoid function

  // TODO: Add code to print the result to the serial monitor
  float y_pred = predict(X_scaled);

    Serial.print("Humidity: ");
  Serial.print(h);
  Serial.print("%  Temperature: ");
  Serial.print(t);
  Serial.print("°C  |  Predicted Probability: ");
  Serial.println(y_pred, 4);

  if (y_pred >= 0.5)
    Serial.println("Predicted Class: 1");
  else
    Serial.println("Predicted Class: 0");

  // Compute heat index in Fahrenheit (the default)
  // float hif = dht.computeHeatIndex(f, h);
  // Compute heat index in Celsius (isFahreheit = false)
  // float hic = dht.computeHeatIndex(t, h, false);

  
  // Serial.print(F("°F  Heat index: "));
  // Serial.print(hic);
  // Serial.print(F("°C "));
  // Serial.print(hif);
  // Serial.println(F("°F"));
}