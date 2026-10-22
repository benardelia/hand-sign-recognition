import 'package:flutter/material.dart';

import 'src/settings.dart';
import 'src/ui/home_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = await AppSettings.load();
  runApp(ScribeSignApp(settings: settings));
}

class ScribeSignApp extends StatelessWidget {
  const ScribeSignApp({super.key, required this.settings});

  final AppSettings settings;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ScribeSign',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF8B5CF6), brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: HomePage(settings: settings),
    );
  }
}
