#include <rn2xx3.h>
#include <SoftwareSerial.h>

/* Copy and fill in the lines from TTN Console -> Devices -> Overview tab -> "EXAMPLE CODE"
.. And add this to a file in the same folder as this sketch: */
#include "ttn-config.h"

const char *devAddr = config_devAddr;
const char *nwkSKey = config_nwkSKey;
const char *appSKey = config_appSKey;

SoftwareSerial mySerial(7, 8); // RX, TX
#define RST  2

// Prepare for receiving data
const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

rn2xx3 myLora(mySerial);

// Setup routine runs once when you press reset
void setup() {
  pinMode(13, OUTPUT);
  led_on();

  // Open serial communications and wait for port to open:
  Serial.begin(9600);
  mySerial.begin(9600);
  Serial.println("Startup");

  // Reset rn2483
  pinMode(RST, OUTPUT);
  digitalWrite(RST, HIGH);
  digitalWrite(RST, LOW);
  delay(500);
  digitalWrite(RST, HIGH);

  // Initialise the rn2483 module
  myLora.autobaud();

  Serial.println("When using OTAA, register this DevEUI: ");
  Serial.println(myLora.hweui());
  Serial.print("RN2483 version number: ");
  Serial.println(myLora.sysver());

  myLora.initABP(devAddr, appSKey, nwkSKey);

  led_off();
  delay(2000);
}


// transmit the average values to TTN
void sendLora(uint16_t value) {
  led_on();
  byte payload[2];
  payload[0] = highByte(value);
  payload[1] = lowByte(value);
  Serial.println("Transmitting:");
  Serial.println(value);
  myLora.txBytes(payload, sizeof(payload));
  led_off();
}

int cycle = 0;
int loudCount = 0;
float loudAvg = 0;

// the loop routine runs over and over again forever:
void loop() {
    recvWithEndMarker();
    showNewNumber();
}

void led_on()
{
  digitalWrite(13, 1);
}

void led_off()
{
  digitalWrite(13, 0);
}

void recvWithEndMarker() {
    static byte ndx = 0;
    char endMarker = '\n';
    char rc;
   
    if (Serial.available() > 0) {
        rc = Serial.read();

        if (rc != endMarker) {
            receivedChars[ndx] = rc;
            ndx++;
            if (ndx >= numChars) {
                ndx = numChars - 1;
            }
        }
        else {
            receivedChars[ndx] = '\0'; // terminate the string
            ndx = 0;
            newData = true;
        }
    }
}

void showNewNumber() {
    if (newData == true) {
        int dataNumber = 0;             // new for this version
        dataNumber = atoi(receivedChars);   // new for this version
        Serial.print("This just in ... ");
        Serial.println(receivedChars);
        Serial.print("Data as Number ... ");    // new for this version
        Serial.println(dataNumber);     // new for this version
        newData = false;
        sendLora((uint16_t)5);
    }
}
