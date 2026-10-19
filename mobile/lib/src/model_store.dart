import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// MediaPipe holistic model (same file as the browser uses).
const holisticModelUrl = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/'
    'holistic_landmarker/float16/latest/holistic_landmarker.task';

/// A trained profile from the hub (`Model/<profile>/` on the server).
class ProfileModel {
  ProfileModel({
    required this.name,
    required this.version,
    required this.labels,
    required this.sequenceLength,
    required this.cameraWidth,
    required this.cameraFps,
    required this.tflitePath,
  });

  final String name;
  final String version;
  final List<String> labels;
  final int sequenceLength;
  final int cameraWidth;
  final int cameraFps;
  final String tflitePath;

  static ProfileModel fromMeta(Map<String, dynamic> meta, String tflitePath) {
    final camera = meta['camera'] as Map<String, dynamic>;
    return ProfileModel(
      name: meta['name'] as String,
      version: meta['version'] as String,
      labels: (meta['labels'] as List).cast<String>(),
      sequenceLength: meta['sequence_length'] as int,
      cameraWidth: camera['width'] as int,
      cameraFps: camera['fps'] as int,
      tflitePath: tflitePath,
    );
  }
}

typedef ProgressCallback = void Function(String message, double? progress);

/// Downloads and caches the MediaPipe model and the hub's profile models.
/// Everything is kept on disk, so the app works offline after the first sync.
class ModelStore {
  ModelStore({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  Future<Directory> _root() async {
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/models');
    return dir.create(recursive: true);
  }

  Future<String> ensureHolisticModel({ProgressCallback? onProgress}) async {
    final file = File('${(await _root()).path}/holistic_landmarker.task');
    if (await file.exists() && await file.length() > 1000000) return file.path;
    await _download(Uri.parse(holisticModelUrl), file, 'Downloading hand/body tracking model', onProgress);
    return file.path;
  }

  /// Returns the newest model for [profile]: downloads it when the hub has a newer
  /// version, otherwise uses the cached copy (also when the hub is unreachable).
  Future<ProfileModel> ensureProfile(String serverUrl, String profile, {ProgressCallback? onProgress}) async {
    final dir = await Directory('${(await _root()).path}/$profile').create(recursive: true);
    final metaFile = File('${dir.path}/profile.json');
    final tflite = File('${dir.path}/model.tflite');
    Map<String, dynamic>? local;
    if (await metaFile.exists() && await tflite.exists()) {
      local = jsonDecode(await metaFile.readAsString()) as Map<String, dynamic>;
    }

    try {
      onProgress?.call('Checking for model updates', null);
      final res = await _client.get(Uri.parse('$serverUrl/api/models')).timeout(const Duration(seconds: 8));
      if (res.statusCode != 200) throw HttpException('GET /api/models: ${res.statusCode}');
      final remote = (jsonDecode(res.body)['profiles'] as Map<String, dynamic>)[profile] as Map<String, dynamic>?;
      if (remote == null) {
        if (local != null) return ProfileModel.fromMeta(local, tflite.path);
        throw StateError("The server has no trained '$profile' model yet. Train it in the web dashboard.");
      }
      final files = (remote['files'] as Map<String, dynamic>).cast<String, String>();
      if (!files.containsKey('model.tflite')) {
        throw StateError("Profile '$profile' has no mobile (TFLite) model.");
      }
      if (local == null || local['version'] != remote['version']) {
        await _download(Uri.parse('$serverUrl${files['model.tflite']}'), tflite, 'Downloading $profile model', onProgress);
        final metaRes = await _client.get(Uri.parse('$serverUrl${files['profile.json']}'));
        if (metaRes.statusCode != 200) throw HttpException('profile.json: ${metaRes.statusCode}');
        await metaFile.writeAsString(metaRes.body);
        local = jsonDecode(metaRes.body) as Map<String, dynamic>;
      }
    } on StateError {
      rethrow;
    } catch (e) {
      if (local == null) {
        throw StateError('Cannot reach the server ($serverUrl) and no cached model exists: $e');
      }
      onProgress?.call('Offline: using cached model', null);
    }
    return ProfileModel.fromMeta(local, tflite.path);
  }

  Future<void> _download(Uri url, File target, String label, ProgressCallback? onProgress) async {
    final res = await _client.send(http.Request('GET', url));
    if (res.statusCode != 200) throw HttpException('Download failed (${res.statusCode}): $url');
    final total = res.contentLength ?? 0;
    final builder = BytesBuilder(copy: false);
    var received = 0;
    await for (final chunk in res.stream) {
      builder.add(chunk);
      received += chunk.length;
      onProgress?.call(label, total > 0 ? received / total : null);
    }
    final tmp = File('${target.path}.part');
    await tmp.writeAsBytes(builder.takeBytes(), flush: true);
    await tmp.rename(target.path);
  }
}
