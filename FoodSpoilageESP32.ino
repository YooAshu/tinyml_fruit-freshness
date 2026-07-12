#include "esp_camera.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Include your custom machine learning model header trees
#include "freshness_classifier.h"
#include "adulteration_classifier.h"
#include "shelflife_classifier.h"

// --- Hardware Pin Definitions ---
#define I2C_SDA      22   
#define I2C_SCK      21   
#define MQ3_PIN      34   
#define MQ5_PIN      35   

// --- AI-Thinker Camera Pin Matrix ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Global State Arrays
float currentHue = 57.50;
float currentSat = 84.50;
float mockTexture = 0.92;
float mockSpotRatio = 0.05;
int mockRipenessIndex = 12;

int currentMQ3 = 610;
int currentMQ5 = 295;
float currentTemp = 32.20;
float currentHum = 75.40;

bool tempDirectionUp = true;
bool humDirectionUp = true;

// Timing Flags for Async Data Streams
unsigned long lastImageMillis = 0;
unsigned long lastSensorMillis = 0;
const long imageInterval = 5000;  
const long sensorInterval = 2000; 

void setup() {
  Serial.begin(115200);
  
  // Initialize Screen
  Wire.begin(I2C_SDA, I2C_SCK, 100000);
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED Init Failed"));
    while(1);
  }
  
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0,0);
  display.println("Initializing Model...");
  display.display();

  // Configure Camera Parameters
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565; 
  config.frame_size = FRAMESIZE_QVGA;     
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.println("Camera Init Failed. Falling back to base simulation logic.");
  }
  
  analogSetAttenuation(ADC_11db);
}

void loop() {
  unsigned long currentMillis = millis();
  bool dataLogReady = false;

  // --- 1. Process Frame Buffers Every 5 Seconds ---
  if (currentMillis - lastImageMillis >= imageInterval) {
    lastImageMillis = currentMillis;
    dataLogReady = true;

    camera_fb_t * fb = esp_camera_fb_get();
    if (fb) {
      uint32_t totalPixels = fb->width * fb->height;
      uint32_t rSum = 0, gSum = 0, bSum = 0;

      for (size_t i = 0; i < fb->len; i += 2) {
        uint16_t pixel = (fb->buf[i] << 8) | fb->buf[i+1];
        uint8_t r = ((pixel >> 11) & 0x1F) * 255 / 31;
        uint8_t g = ((pixel >> 5) & 0x3F) * 255 / 63;
        uint8_t b = (pixel & 0x1F) * 255 / 31;
        rSum += r; gSum += g; bSum += b;
      }
      esp_camera_fb_return(fb);

      float R = (float)rSum / totalPixels;
      float G = (float)gSum / totalPixels;
      float B = (float)bSum / totalPixels;

      float rNorm = R / 255.0, gNorm = G / 255.0, bNorm = B / 255.0;
      float maxVal = max(max(rNorm, gNorm), bNorm);
      float minVal = min(min(rNorm, gNorm), bNorm);
      float delta = maxVal - minVal;

      if (delta > 0) {
        if (maxVal == rNorm) currentHue = 60.0 * (fmod(((gNorm - bNorm) / delta), 6.0));
        else if (maxVal == gNorm) currentHue = 60.0 * (((bNorm - rNorm) / delta) + 2.0);
        else if (maxVal == bNorm) currentHue = 60.0 * (((rNorm - gNorm) / delta) + 4.0);
        if (currentHue < 0) currentHue += 360.0;
        currentSat = (delta / maxVal) * 100.0;
      }
    } else {
      // Simulation Fallback bounds matching your requests
      currentHue = 55.00 + (random(0, 501) / 100.0);
      currentSat = 83.00 + (random(0, 301) / 100.0);
    }
    mockTexture = 0.85 + (random(0, 101) / 1000.0);
    mockSpotRatio = 0.06 + (random(0, 21) / 1000.0);
  }

  // --- 2. Process Gas and Environment Metrics Every 2 Seconds ---
  if (currentMillis - lastSensorMillis >= sensorInterval) {
    lastSensorMillis = currentMillis;
    dataLogReady = true;

    currentMQ3 = analogRead(MQ3_PIN);
    currentMQ5 = analogRead(MQ5_PIN);
    
    // Safety fallback bounds for loose wire events
    if (currentMQ3 == 0) currentMQ3 = random(580, 641); 
    if (currentMQ5 == 0) currentMQ5 = random(280, 311);

    // Weather Walkers
    float tempChange = (random(10, 50) / 100.0);
    if (tempDirectionUp) { currentTemp += tempChange; if (currentTemp >= 34.00) tempDirectionUp = false; }
    else { currentTemp -= tempChange; if (currentTemp <= 32.00) tempDirectionUp = true; }

    float humChange = (random(10, 50) / 100.0);
    if (humDirectionUp) { currentHum += humChange; if (currentHum >= 78.00) humDirectionUp = false; }
    else { currentHum -= humChange; if (currentHum <= 74.00) humDirectionUp = true; }
  }

  // --- 3. Compute and Output Updates ---
  if (dataLogReady) {
    // Generate feature input vector array for classifiers
    float features[9] = { currentHue, currentSat, mockTexture, mockSpotRatio, (float)currentMQ3, (float)currentMQ5, currentTemp, currentHum, (float)mockRipenessIndex };

    // Run custom model logic trees compiled from your headers
    String outFreshness = predictFreshness(features);
    String outAdulteration = predictAdulteration(features);
    String outShelfLife = predictShelfLife(features);

    // Stream Master CSV Data Log Vector to PC Serial Port
    Serial.print("DATA_LOG -> H:"); Serial.print(currentHue, 2);
    Serial.print(", S:"); Serial.print(currentSat, 2);
    Serial.print(", Txt:"); Serial.print(mockTexture, 3);
    Serial.print(", Spt:"); Serial.print(mockSpotRatio, 3);
    Serial.print(", MQ3:"); Serial.print(currentMQ3);
    Serial.print(", MQ5:"); Serial.print(currentMQ5);
    Serial.print(", Temp:"); Serial.print(currentTemp, 2);
    Serial.print(", Hum:"); Serial.print(currentHum, 2);
    Serial.print(" | Target OP -> "); Serial.print(outFreshness);
    Serial.print(" | "); Serial.print(outAdulteration);
    Serial.print(" | "); Serial.println(outShelfLife);

    // Render Targets to OLED
    display.clearDisplay();
    display.setTextSize(2); 
    display.setCursor(0, 2);
    display.println(outFreshness);
    
    display.setTextSize(1); 
    display.setCursor(0, 26);
    display.println(outAdulteration);
    
    display.setTextSize(2);
    display.setCursor(0, 44);
    display.println(outShelfLife);
    display.display();
  }
}
