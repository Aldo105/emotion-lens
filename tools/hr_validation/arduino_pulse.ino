/*
 * EmotionLens — referencia de pulso para validar el rPPG
 *
 * Lee un MAX30102/MAX30105 por I2C y emite una linea por latido:
 *     millis,bpm,bpm_promedio
 *
 * El firmware se mantiene deliberadamente tonto: solo mide y emite. Todo el
 * analisis y la sincronizacion se hacen en el PC, donde es facil corregir un
 * error sin volver a flashear.
 *
 * Librería necesaria (Gestor de librerías del IDE de Arduino):
 *     "SparkFun MAX3010x Pulse and Proximity Sensor Library"
 *
 * Conexion (Arduino Uno/Nano):
 *     VIN -> 3.3V     GND -> GND     SDA -> A4     SCL -> A5
 * En un Leonardo/Micro: SDA -> D2, SCL -> D3.
 */

#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

MAX30105 sensor;

const byte VENTANA = 8;          // latidos promediados
byte tasas[VENTANA];
byte indiceTasa = 0;
long ultimoLatido = 0;
float bpmPromedio = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("# ERROR: MAX3010x no encontrado. Revisa el cableado I2C.");
    while (1) { ; }
  }

  // Configuracion recomendada para deteccion de latido por dedo
  sensor.setup();
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeGreen(0);

  Serial.println("# listo: coloca el dedo sobre el sensor");
  Serial.println("# millis,bpm,bpm_promedio");
}

void loop() {
  long ir = sensor.getIR();

  // Por debajo de este umbral no hay dedo puesto
  if (ir < 50000) {
    delay(50);
    return;
  }

  if (checkForBeat(ir)) {
    long ahora = millis();
    long delta = ahora - ultimoLatido;
    ultimoLatido = ahora;

    float bpm = 60.0 / (delta / 1000.0);

    if (bpm > 40 && bpm < 200) {
      tasas[indiceTasa++] = (byte)bpm;
      indiceTasa %= VENTANA;

      int suma = 0;
      for (byte i = 0; i < VENTANA; i++) suma += tasas[i];
      bpmPromedio = suma / (float)VENTANA;

      Serial.print(ahora);
      Serial.print(",");
      Serial.print(bpm, 1);
      Serial.print(",");
      Serial.println(bpmPromedio, 1);
    }
  }
}
