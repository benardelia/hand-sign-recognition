import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_reactive_ble/flutter_reactive_ble.dart';

/// BLE link to the glasses firmware (firmware/glasses_esp32s3). The glasses
/// advertise [serviceUuid]; the app writes translated text to [textUuid] and
/// reads the camera stream URL from [infoUuid].
class GlassesLink extends ChangeNotifier {
  static final serviceUuid = Uuid.parse('a7d3e100-5c4b-4c1e-9f5a-2b8e6c0d9a01');
  static final textUuid = Uuid.parse('a7d3e101-5c4b-4c1e-9f5a-2b8e6c0d9a01');
  static final infoUuid = Uuid.parse('a7d3e102-5c4b-4c1e-9f5a-2b8e6c0d9a01');
  static const maxTextBytes = 180;

  final _ble = FlutterReactiveBle();
  StreamSubscription<ConnectionStateUpdate>? _connection;
  QualifiedCharacteristic? _text;
  String status = 'Not connected';
  String? streamUrl;
  String _lastSent = '';

  bool get connected => _text != null;

  Future<void> connect() async {
    if (connected) return;
    final bleStatus = await _ble.statusStream.firstWhere((s) => s != BleStatus.unknown);
    if (bleStatus != BleStatus.ready) {
      _setStatus('Bluetooth not ready ($bleStatus)');
      return;
    }

    _setStatus('Scanning for glasses...');
    DiscoveredDevice? device;
    try {
      device = await _ble
          .scanForDevices(withServices: [serviceUuid], requireLocationServicesEnabled: false)
          .first
          .timeout(const Duration(seconds: 10));
    } on TimeoutException {
      _setStatus('Glasses not found');
      return;
    }

    _setStatus('Connecting...');
    final ready = Completer<void>();
    await _connection?.cancel();
    _connection = _ble
        .connectToDevice(id: device.id, connectionTimeout: const Duration(seconds: 10))
        .listen((update) {
      if (update.connectionState == DeviceConnectionState.connected && !ready.isCompleted) {
        ready.complete();
      } else if (update.connectionState == DeviceConnectionState.disconnected) {
        _text = null;
        if (!ready.isCompleted) ready.completeError(StateError('Connection failed'));
        _setStatus('Disconnected');
      }
    }, onError: (Object e) {
      if (!ready.isCompleted) ready.completeError(e);
      _text = null;
      _setStatus('Connection error');
    });
    await ready.future;

    if (defaultTargetPlatform == TargetPlatform.android) {
      await _ble.requestMtu(deviceId: device.id, mtu: maxTextBytes + 5);
    }
    _text = QualifiedCharacteristic(serviceId: serviceUuid, characteristicId: textUuid, deviceId: device.id);
    try {
      final info = await _ble.readCharacteristic(
          QualifiedCharacteristic(serviceId: serviceUuid, characteristicId: infoUuid, deviceId: device.id));
      streamUrl = (jsonDecode(utf8.decode(info)) as Map<String, dynamic>)['stream'] as String?;
    } catch (_) {
      streamUrl = null;
    }
    _lastSent = '';
    _setStatus('Glasses connected');
  }

  /// Shows [text] on the glasses display. Repeated text is not resent.
  Future<void> send(String text) async {
    final c = _text;
    if (c == null || text == _lastSent) return;
    _lastSent = text;
    var bytes = utf8.encode(text);
    if (bytes.length > maxTextBytes) {
      // Keep the newest words, without cutting a multi-byte character in half.
      var start = bytes.length - maxTextBytes;
      while (start < bytes.length && (bytes[start] & 0xC0) == 0x80) {
        start++;
      }
      bytes = bytes.sublist(start);
    }
    try {
      await _ble.writeCharacteristicWithResponse(c, value: bytes);
    } catch (e) {
      _setStatus('Send failed: $e');
    }
  }

  Future<void> disconnect() async {
    await _connection?.cancel();
    _connection = null;
    _text = null;
    _setStatus('Not connected');
  }

  void _setStatus(String s) {
    status = s;
    notifyListeners();
  }
}
