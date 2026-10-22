// ScribeSign glasses firmware configuration.
#pragma once

// ---- Board (camera pin map, see camera_pins.h) ----
// CAMERA_MODEL_XIAO_ESP32S3 : Seeed Studio XIAO ESP32S3 Sense (default, small enough for glasses)
// CAMERA_MODEL_ESP32S3_EYE  : Espressif ESP32-S3-EYE
#if !defined(CAMERA_MODEL_XIAO_ESP32S3) && !defined(CAMERA_MODEL_ESP32S3_EYE)
#define CAMERA_MODEL_XIAO_ESP32S3
#endif

// ---- Wi-Fi ----
// Access point (default): the phone joins this network and reads http://192.168.4.1:81/stream.
// Station: set WIFI_STA_SSID to join an existing network / the phone's hotspot instead;
// the stream URL is then sent to the app over Bluetooth.
#define WIFI_AP_SSID "ScribeSign-Glasses"
#define WIFI_AP_PASSWORD "scribesign"  // at least 8 characters
#define WIFI_STA_SSID ""
#define WIFI_STA_PASSWORD ""

// ---- Camera ----
// VGA (640x480) covers both phone profiles: LOW downscales to 480 px, HIGH uses 640 px.
#define CAMERA_FRAME_SIZE FRAMESIZE_VGA
#define CAMERA_JPEG_QUALITY 12  // 0-63, lower = better quality / bigger frames
#define STREAM_PORT 81

// ---- Display: SSD1306 128x64 OLED on I2C ----
#define DISPLAY_SDA 5  // XIAO ESP32S3 D4
#define DISPLAY_SCL 6  // XIAO ESP32S3 D5
#define DISPLAY_FLIP 0  // 1 = rotate 180 degrees (depends on how the display is mounted)

// ---- Bluetooth (must match mobile/lib/src/glasses_link.dart) ----
#define BLE_DEVICE_NAME "ScribeSign Glasses"
#define BLE_SERVICE_UUID "a7d3e100-5c4b-4c1e-9f5a-2b8e6c0d9a01"
#define BLE_TEXT_UUID "a7d3e101-5c4b-4c1e-9f5a-2b8e6c0d9a01"  // app writes the sentence (UTF-8)
#define BLE_INFO_UUID "a7d3e102-5c4b-4c1e-9f5a-2b8e6c0d9a01"  // app reads {"stream": "..."}
