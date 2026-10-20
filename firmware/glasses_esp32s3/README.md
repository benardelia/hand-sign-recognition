# Glasses firmware (ESP32-S3)

The glasses only **capture video** and **show text**. All recognition runs in the phone
app (`mobile/`).

| Function | How |
|---|---|
| Camera | MJPEG stream at `http://<ip>:81/stream` (VGA, JPEG quality 12) |
| Text display | 128x64 SSD1306 OLED; the newest 5 word-wrapped lines of the sentence |
| Link to phone | Bluetooth LE: the app writes the sentence and reads the stream URL |

## Hardware

- **Board:** Seeed Studio XIAO ESP32S3 Sense (camera + 8 MB PSRAM, 21 x 17.5 mm). The
  ESP32-S3-EYE also works (`CAMERA_MODEL_ESP32S3_EYE` in `config.h`).
- **Display:** 0.96" or 0.91" SSD1306 I2C OLED. For a heads-up view, mount it facing a
  small beam-splitter/prism in front of one lens.
- **Power:** 3.7 V LiPo on the XIAO battery pads (it has a charger built in).

Wiring (XIAO ESP32S3 Sense):

| OLED | XIAO |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | D4 (GPIO5) |
| SCL | D5 (GPIO6) |

Mount the camera so it sees the person **in front of the wearer** (the signer). It does
not need to see the wearer's own hands.

## Build and flash

Arduino IDE 2 or `arduino-cli`, with:

- Board package **esp32 by Espressif** 3.x, board **XIAO_ESP32S3**
  (Tools → PSRAM: **OPI PSRAM**)
- Libraries: **NimBLE-Arduino** 2.x, **U8g2**

```bash
arduino-cli core install esp32:esp32
arduino-cli lib install "NimBLE-Arduino" "U8g2"
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi firmware/glasses_esp32s3
arduino-cli upload  --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi -p <port> firmware/glasses_esp32s3
```

ESP32-S3-EYE (generic ESP32-S3 board, larger app partition):

```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3:PSRAM=opi,PartitionScheme=huge_app,FlashSize=8M \
  --build-property "compiler.cpp.extra_flags=-DCAMERA_MODEL_ESP32S3_EYE" firmware/glasses_esp32s3
```

Both builds compile with esp32 core 3.3.12, NimBLE-Arduino 2.5.1 and U8g2 2.36.19
(XIAO: 1.18 MB flash, 66 KB static RAM).

## Using it with the app

1. Power the glasses. The OLED shows `Join ScribeSign-Glasses` and the stream URL.
2. On the phone, join the `ScribeSign-Glasses` Wi-Fi (password `scribesign`, set in
   `config.h`). Download the models in the app first, because this network has no internet.
3. In the app: **Settings → Camera → Glasses**, then tap the Bluetooth icon. The app finds
   the glasses, reads the stream URL and starts translating. Sentences appear on the OLED.

To keep internet on the phone, set `WIFI_STA_SSID` / `WIFI_STA_PASSWORD` in `config.h` to
the phone's hotspot. The glasses then join it, and the app gets the stream URL over
Bluetooth.

## Bluetooth interface

| UUID | Access | Content |
|---|---|---|
| `a7d3e100-5c4b-4c1e-9f5a-2b8e6c0d9a01` | service | |
| `a7d3e101-5c4b-4c1e-9f5a-2b8e6c0d9a01` | write | sentence, UTF-8, up to 180 bytes |
| `a7d3e102-5c4b-4c1e-9f5a-2b8e6c0d9a01` | read | `{"stream": "http://<ip>:81/stream"}` |
