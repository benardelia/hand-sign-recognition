import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum InputSource { phone, glasses }

/// User settings, persisted with SharedPreferences.
class AppSettings extends ChangeNotifier {
  AppSettings._(this._prefs);

  static const defaultServerUrl =
      'http://192.168.0.3:5001'; // 'https://handsign.duckdns.org';
  static const defaultStreamUrl = 'http://192.168.4.1:81/stream';

  final SharedPreferences _prefs;

  static Future<AppSettings> load() async =>
      AppSettings._(await SharedPreferences.getInstance());

  /// Flask hub that serves trained models and receives samples.
  String get serverUrl => _prefs.getString('serverUrl') ?? defaultServerUrl;
  set serverUrl(String v) =>
      _set('serverUrl', v.trim().replaceAll(RegExp(r'/+$'), ''));

  /// Quality profile (config/profiles.json): 'low' for glasses / mid-range phones, 'high' for flagships.
  String get profile => _prefs.getString('profile') ?? 'low';
  set profile(String v) => _set('profile', v);

  InputSource get source => InputSource.values.byName(
    _prefs.getString('source') ?? InputSource.phone.name,
  );
  set source(InputSource v) => _set('source', v.name);

  /// MJPEG stream of the glasses camera; filled automatically from the glasses over BLE when available.
  String get glassesStreamUrl =>
      _prefs.getString('glassesStreamUrl') ?? defaultStreamUrl;
  set glassesStreamUrl(String v) => _set('glassesStreamUrl', v.trim());

  /// Clockwise rotation applied to camera frames; -1 = automatic (sensor orientation).
  int get rotationOverride => _prefs.getInt('rotationOverride') ?? -1;
  set rotationOverride(int v) => _set('rotationOverride', v);

  bool get useFrontCamera => _prefs.getBool('useFrontCamera') ?? false;
  set useFrontCamera(bool v) => _set('useFrontCamera', v);

  double get threshold => _prefs.getDouble('threshold') ?? 0.85;
  set threshold(double v) => _set('threshold', v);

  bool get useGpu => _prefs.getBool('useGpu') ?? true;
  set useGpu(bool v) => _set('useGpu', v);

  /// Optional X-Handsign-Token for servers that protect sample uploads.
  String get collectToken => _prefs.getString('collectToken') ?? '';
  set collectToken(String v) => _set('collectToken', v.trim());

  void _set(String key, Object value) {
    switch (value) {
      case String s:
        _prefs.setString(key, s);
      case int i:
        _prefs.setInt(key, i);
      case double d:
        _prefs.setDouble(key, d);
      case bool b:
        _prefs.setBool(key, b);
    }
    notifyListeners();
  }
}
