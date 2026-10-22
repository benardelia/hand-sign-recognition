import 'package:flutter/material.dart';

import '../settings.dart';
import 'context_menu.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key, required this.settings});

  final AppSettings settings;

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late final _server = TextEditingController(text: widget.settings.serverUrl);
  late final _stream = TextEditingController(text: widget.settings.glassesStreamUrl);
  late final _token = TextEditingController(text: widget.settings.collectToken);

  AppSettings get s => widget.settings;

  @override
  void dispose() {
    s.serverUrl = _server.text;
    s.glassesStreamUrl = _stream.text;
    s.collectToken = _token.text;
    _server.dispose();
    _stream.dispose();
    _token.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) => Scaffold(
        appBar: AppBar(title: const Text('Settings')),
        body: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const _Header('Quality profile'),
            SegmentedButton<String>(
              segments: const [
                ButtonSegment(value: 'low', label: Text('LOW'), icon: Icon(Icons.speed)),
                ButtonSegment(value: 'high', label: Text('HIGH'), icon: Icon(Icons.high_quality)),
              ],
              selected: {s.profile},
              onSelectionChanged: (v) => s.profile = v.first,
            ),
            const Padding(
              padding: EdgeInsets.only(top: 6),
              child: Text('LOW: hands + upper body, 15 fps, smallest model (glasses, mid-range phones).\n'
                  'HIGH: adds lips/eyebrows, 30 fps (flagship phones).'),
            ),
            const _Header('Camera'),
            SegmentedButton<InputSource>(
              segments: const [
                ButtonSegment(value: InputSource.phone, label: Text('Phone'), icon: Icon(Icons.smartphone)),
                ButtonSegment(value: InputSource.glasses, label: Text('Glasses'), icon: Icon(Icons.visibility)),
              ],
              selected: {s.source},
              onSelectionChanged: (v) => s.source = v.first,
            ),
            if (s.source == InputSource.phone)
              SwitchListTile(
                title: const Text('Use front camera'),
                subtitle: const Text('Back camera = point the phone at the signer'),
                value: s.useFrontCamera,
                onChanged: (v) => s.useFrontCamera = v,
              ),
            if (s.source == InputSource.glasses)
              TextField(
                contextMenuBuilder: flutterContextMenu,
                controller: _stream,
                decoration: const InputDecoration(
                  labelText: 'Glasses stream URL',
                  helperText: 'Filled automatically when the glasses are connected over Bluetooth',
                ),
                keyboardType: TextInputType.url,
              ),
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Frame rotation'),
              subtitle: const Text('Change if the landmarks do not line up with the person'),
              trailing: DropdownButton<int>(
                value: s.rotationOverride,
                items: const [
                  DropdownMenuItem(value: -1, child: Text('Auto')),
                  DropdownMenuItem(value: 0, child: Text('0°')),
                  DropdownMenuItem(value: 90, child: Text('90°')),
                  DropdownMenuItem(value: 180, child: Text('180°')),
                  DropdownMenuItem(value: 270, child: Text('270°')),
                ],
                onChanged: (v) => s.rotationOverride = v ?? -1,
              ),
            ),
            const _Header('Recognition'),
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: Text('Confidence threshold: ${(s.threshold * 100).round()}%'),
              subtitle: Slider(
                value: s.threshold,
                min: 0.5,
                max: 0.99,
                onChanged: (v) => s.threshold = v,
              ),
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('GPU acceleration'),
              subtitle: const Text('Falls back to CPU automatically'),
              value: s.useGpu,
              onChanged: (v) => s.useGpu = v,
            ),
            const _Header('Server (model updates & data collection)'),
            TextField(
              contextMenuBuilder: flutterContextMenu,
              controller: _server,
              decoration: const InputDecoration(labelText: 'Server URL'),
              keyboardType: TextInputType.url,
            ),
            TextField(
              contextMenuBuilder: flutterContextMenu,
              controller: _token,
              decoration: const InputDecoration(labelText: 'Upload token (optional)'),
              obscureText: true,
            ),
          ],
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header(this.text);

  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 20, bottom: 8),
        child: Text(text, style: Theme.of(context).textTheme.titleMedium),
      );
}
