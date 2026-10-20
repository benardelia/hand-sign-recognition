// ScribeSign smart glasses: camera + display, no machine learning on the glasses.
//
//  - Streams the camera as MJPEG over Wi-Fi:  http://<ip>:81/stream
//  - Bluetooth LE service: the phone app writes the translated sentence, which is
//    word-wrapped on a 128x64 OLED; the app reads the stream URL from the info characteristic.
//
// The phone (mobile/ Flutter app) does landmark tracking and sign classification.
// Board: ESP32-S3 with PSRAM (XIAO ESP32S3 Sense / ESP32-S3-EYE), Arduino-ESP32 core 3.x.
// Libraries: NimBLE-Arduino 2.x, U8g2.

#include <Arduino.h>
#include <NimBLEDevice.h>
#include <U8g2lib.h>
#include <WiFi.h>
#include <Wire.h>

#include "config.h"
#include "camera_pins.h"
#include "esp_camera.h"
#include "esp_http_server.h"

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------
U8G2_SSD1306_128X64_NONAME_F_HW_I2C display(DISPLAY_FLIP ? U8G2_R2 : U8G2_R0, U8X8_PIN_NONE);

static const int kLineHeight = 12;
static const int kMaxLines = 5;
static const int kMaxChars = 21;  // 128 px / 6 px font

// Written from the NimBLE task, read in loop(). A mutex (not a critical section): String
// copies allocate memory, which must not happen with interrupts disabled.
SemaphoreHandle_t textLock;
String pendingText;
bool textChanged = false;

static void setPendingText(const String &text) {
  xSemaphoreTake(textLock, portMAX_DELAY);
  pendingText = text;
  textChanged = true;
  xSemaphoreGive(textLock);
}
String statusLine = "Starting...";

// Word-wraps text to kMaxChars columns and keeps the newest kMaxLines lines.
static int wrapLines(const String &text, String lines[], int maxLines) {
  String all[32];
  int count = 0;
  String current;
  int start = 0;
  while (start <= (int)text.length() && count < 32) {
    int space = text.indexOf(' ', start);
    String word = text.substring(start, space < 0 ? text.length() : space);
    start = space < 0 ? text.length() + 1 : space + 1;
    while ((int)word.length() > kMaxChars) {  // hard-split very long words
      if (current.length()) { all[count++] = current; current = ""; if (count >= 32) break; }
      all[count++] = word.substring(0, kMaxChars);
      word = word.substring(kMaxChars);
      if (count >= 32) break;
    }
    if (count >= 32) break;
    if (current.length() + word.length() + (current.length() ? 1 : 0) > (unsigned)kMaxChars) {
      all[count++] = current;
      current = word;
    } else {
      current += (current.length() ? " " : "") + word;
    }
  }
  if (current.length() && count < 32) all[count++] = current;
  int first = count > maxLines ? count - maxLines : 0;
  for (int i = first; i < count; i++) lines[i - first] = all[i];
  return count - first;
}

static void drawText(const String &text) {
  String lines[kMaxLines];
  int n = wrapLines(text, lines, kMaxLines);
  display.clearBuffer();
  display.setFont(u8g2_font_6x12_tf);
  for (int i = 0; i < n; i++) display.drawUTF8(0, kLineHeight * (i + 1) - 2, lines[i].c_str());
  display.sendBuffer();
}

// ---------------------------------------------------------------------------
// Camera + MJPEG stream (esp_http_server, as in the CameraWebServer example)
// ---------------------------------------------------------------------------
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *kStreamContentType = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *kStreamBoundary = "\r\n--" PART_BOUNDARY "\r\n";
static const char *kStreamPart = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static bool initCamera() {
  camera_config_t config = {};
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
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = CAMERA_FRAME_SIZE;
  config.jpeg_quality = CAMERA_JPEG_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;  // always stream the newest frame
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

static esp_err_t streamHandler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, kStreamContentType);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  char part[64];
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }
    size_t len = snprintf(part, sizeof(part), kStreamPart, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, kStreamBoundary, strlen(kStreamBoundary));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part, len);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;  // client disconnected
  }
  return res;
}

static void startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = STREAM_PORT;
  config.ctrl_port = STREAM_PORT + 32768;
  httpd_handle_t server = nullptr;
  httpd_uri_t uri = {};
  uri.uri = "/stream";
  uri.method = HTTP_GET;
  uri.handler = streamHandler;
  if (httpd_start(&server, &config) == ESP_OK) {
    httpd_register_uri_handler(server, &uri);
  }
}

// ---------------------------------------------------------------------------
// Wi-Fi
// ---------------------------------------------------------------------------
String streamUrl;

static void startWifi() {
  bool connected = false;
  if (strlen(WIFI_STA_SSID) > 0) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_STA_SSID, WIFI_STA_PASSWORD);
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
    connected = WiFi.status() == WL_CONNECTED;
  }
  IPAddress ip;
  if (connected) {
    ip = WiFi.localIP();
    statusLine = String("WiFi ") + WIFI_STA_SSID;
  } else {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
    ip = WiFi.softAPIP();
    statusLine = String("Join ") + WIFI_AP_SSID;
  }
  WiFi.setSleep(false);  // lower streaming latency
  streamUrl = "http://" + ip.toString() + ":" + String(STREAM_PORT) + "/stream";
  Serial.println("Stream: " + streamUrl);
}

// ---------------------------------------------------------------------------
// Bluetooth LE (NimBLE-Arduino 2.x)
// ---------------------------------------------------------------------------
class TextCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic *characteristic, NimBLEConnInfo &connInfo) override {
    std::string value = characteristic->getValue();
    setPendingText(String(value.c_str()));
  }
};

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer *server, NimBLEConnInfo &connInfo) override { setPendingText("Phone connected"); }
  void onDisconnect(NimBLEServer *server, NimBLEConnInfo &connInfo, int reason) override {
    setPendingText(statusLine);
    NimBLEDevice::startAdvertising();
  }
};

static void startBle() {
  NimBLEDevice::init(BLE_DEVICE_NAME);
  NimBLEDevice::setMTU(185);
  NimBLEServer *server = NimBLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  NimBLEService *service = server->createService(BLE_SERVICE_UUID);
  NimBLECharacteristic *text =
      service->createCharacteristic(BLE_TEXT_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  text->setCallbacks(new TextCallbacks());
  NimBLECharacteristic *info = service->createCharacteristic(BLE_INFO_UUID, NIMBLE_PROPERTY::READ);
  info->setValue(String("{\"stream\":\"" + streamUrl + "\"}").c_str());
  service->start();
  NimBLEAdvertising *advertising = NimBLEDevice::getAdvertising();
  advertising->addServiceUUID(BLE_SERVICE_UUID);
  advertising->setName(BLE_DEVICE_NAME);
  advertising->start();
}

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  textLock = xSemaphoreCreateMutex();
  Wire.begin(DISPLAY_SDA, DISPLAY_SCL);
  display.begin();
  drawText("ScribeSign starting...");

  if (!psramFound()) Serial.println("Warning: no PSRAM, camera may fail at VGA");
  bool cameraOk = initCamera();
  startWifi();
  if (cameraOk) startStreamServer();
  startBle();
  drawText(cameraOk ? statusLine + " " + streamUrl : "Camera error - check wiring");
}

void loop() {
  String text;
  bool changed = false;
  xSemaphoreTake(textLock, portMAX_DELAY);
  if (textChanged) {
    text = pendingText;
    textChanged = false;
    changed = true;
  }
  xSemaphoreGive(textLock);
  if (changed) drawText(text);
  delay(20);
}
