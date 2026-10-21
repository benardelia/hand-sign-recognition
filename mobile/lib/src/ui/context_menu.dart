import 'package:flutter/material.dart';

/// Flutter-drawn copy/paste menu for text fields.
///
/// The default on iOS is the native system menu (SystemContextMenu), which asserts
/// "Attempted to show while another instance was still visible" when focus moves
/// between text fields while a menu is open. Flutter's own toolbar has no such limit.
Widget flutterContextMenu(
  BuildContext context,
  EditableTextState editableTextState,
) => AdaptiveTextSelectionToolbar.editableText(
  editableTextState: editableTextState,
);
