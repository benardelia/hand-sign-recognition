"""
Train a sign classifier for one quality profile (see config/profiles.json).

    python train_model.py --profile high
    python train_model.py --profile all

Every profile trains on the same raw holistic dataset (Data/Holistic_Landmarks,
(T, 1662) .npy sequences). Output goes to Model/<profile>/:
    model.keras         Keras model (raw (T, 1662) input, preprocessing inside)
    model.tflite        mobile model (profiles with export.tflite = true)
    labels.txt          one label per line, index = class id
    profile.json        the profile config + training metadata
    feature_spec.json   landmark selection / normalization description
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, EarlyStopping

from utils.features import (HOLISTIC_SIZE, augment_sequence, build_feature_spec, get_profile,
                            load_profiles, make_preprocess_layer, resample_sequence, select_features)
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

DATA_PATH = os.path.join('Data', 'Holistic_Landmarks')
MODELS_PATH = 'Model'


class WebProgressCallback(Callback):
    """Prints a machine-readable line per epoch; parsed by app.py for the web dashboard."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEPOCH_END:{epoch+1}:{logs.get('categorical_accuracy', 0.0):.4f}:{logs.get('loss', 0.0):.4f}:"
              f"{logs.get('val_categorical_accuracy', 0.0):.4f}:{logs.get('val_loss', 0.0):.4f}", flush=True)


def load_dataset(data_path, sequence_length):
    labels = sorted(d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))) \
        if os.path.exists(data_path) else []
    X, y, skipped = [], [], 0
    for idx, label in enumerate(labels):
        folder = os.path.join(data_path, label)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.npy'):
                continue
            seq = np.load(os.path.join(folder, fname))
            if seq.ndim != 2 or seq.shape[1] != HOLISTIC_SIZE or seq.shape[0] < 2:
                skipped += 1
                continue
            X.append(resample_sequence(seq, sequence_length))
            y.append(idx)
    if skipped:
        logger.warning(f"Skipped {skipped} files that are not (T, {HOLISTIC_SIZE}) holistic sequences.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), labels


def build_model(profile, spec, num_classes):
    t = profile['sequence_length']
    arch = profile['model']['arch']
    inp = layers.Input(shape=(t, HOLISTIC_SIZE), name='holistic_sequence')
    x = make_preprocess_layer(spec)(inp)

    if arch == 'cnn_small':
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
    elif arch == 'cnn':
        x = layers.Dense(128, activation='relu')(x)
        for _ in range(3):
            r = x
            x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Add()([x, r])
        x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
    elif arch == 'lstm':
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(128)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
    else:
        raise ValueError(f"Unknown model arch '{arch}'")

    out = layers.Dense(num_classes, activation='softmax', name='probabilities')(x)
    model = tf.keras.Model(inp, out, name=f"handsign_{profile['name']}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def export_tflite(model, profile, path):
    t = profile['sequence_length']
    run = tf.function(lambda x: model(x, training=False))
    concrete = run.get_concrete_function(tf.TensorSpec([1, t, HOLISTIC_SIZE], tf.float32, name='holistic_sequence'))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    # Builtin ops only: keeps the model loadable by plain LiteRT on Android/iOS (no Flex delegate).
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    quant = profile['export'].get('quantization', 'none')
    if quant in ('dynamic_int8', 'float16'):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quant == 'float16':
        converter.target_spec.supported_types = [tf.float16]
    data = converter.convert()
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def tflite_predict(path, X):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = []
    for sample in X:
        interp.set_tensor(inp['index'], sample[None].astype(np.float32))
        interp.invoke()
        preds.append(interp.get_tensor(out['index'])[0])
    return np.array(preds)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def train_profile(name, data_path=DATA_PATH, models_path=MODELS_PATH, seed=42, epochs=None, batch_size=None):
    profile = get_profile(name)
    profile['model'] = dict(profile['model'])
    if epochs:
        profile['model']['epochs'] = epochs
    if batch_size:
        profile['model']['batch_size'] = batch_size
    spec = build_feature_spec(profile)
    t = profile['sequence_length']
    logger.info(f"--- Training profile '{name}': {spec['num_points']} landmarks, "
                f"{spec['num_features']} features/frame, sequence {t} ---")

    X, y, labels = load_dataset(data_path, t)
    if len(labels) < 2 or len(X) == 0:
        logger.error(f"Need at least 2 labels with holistic samples in {data_path}; found {len(labels)}.")
        return None
    logger.info(f"Labels: {labels} | samples: {len(X)}")

    stratify = y if np.bincount(y).min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=stratify)

    rng = np.random.default_rng(seed)
    aug = profile.get('augment', {})
    X_aug, y_aug = [X_train], [y_train]
    for _ in range(int(aug.get('copies', 0))):
        X_aug.append(np.stack([augment_sequence(s, aug, rng) for s in X_train]))
        y_aug.append(y_train)
    X_train, y_train = np.concatenate(X_aug), np.concatenate(y_aug)
    logger.info(f"Training samples after augmentation: {len(X_train)} | validation: {len(X_val)}")

    num_classes = len(labels)
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    Y_val = tf.keras.utils.to_categorical(y_val, num_classes)

    tf.keras.utils.set_random_seed(seed)
    model = build_model(profile, spec, num_classes)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=profile['model']['epochs'], batch_size=profile['model']['batch_size'],
              callbacks=[WebProgressCallback(),
                         EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)],
              verbose=2)

    val_probs = model.predict(X_val, verbose=0)
    val_acc = float((val_probs.argmax(-1) == y_val).mean())

    out_dir = os.path.join(models_path, name)
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, 'model.keras'))

    metrics = {"val_accuracy": round(val_acc, 4), "num_train": int(len(X_train)), "num_val": int(len(X_val)),
               "params": int(model.count_params())}

    files = {"keras": "model.keras"}
    if profile['export'].get('tflite'):
        tfl_path = os.path.join(out_dir, 'model.tflite')
        size = export_tflite(model, profile, tfl_path)
        tfl_probs = tflite_predict(tfl_path, X_val)
        metrics.update({
            "tflite_bytes": size,
            "tflite_val_accuracy": round(float((tfl_probs.argmax(-1) == y_val).mean()), 4),
            "tflite_max_abs_diff": round(float(np.abs(tfl_probs - val_probs).max()), 5),
            "tflite_sha256": file_sha256(tfl_path),
        })
        files["tflite"] = "model.tflite"
        logger.info(f"TFLite exported: {size / 1024:.1f} KB, accuracy {metrics['tflite_val_accuracy']:.3f}, "
                    f"max diff vs Keras {metrics['tflite_max_abs_diff']}")

    # Sanity check: in-graph preprocessing matches the numpy reference used by tooling/tests.
    pre = tf.keras.Model(model.input, model.get_layer('feature_preprocess').output)
    ref = np.stack([select_features(s, spec) for s in X_val[:4]])
    metrics["preprocess_max_abs_diff"] = round(float(np.abs(pre.predict(X_val[:4], verbose=0) - ref).max()), 6)

    with open(os.path.join(out_dir, 'labels.txt'), 'w') as f:
        f.write("\n".join(labels))
    with open(os.path.join(out_dir, 'feature_spec.json'), 'w') as f:
        json.dump(spec, f)
    version = time.strftime('%Y%m%d-%H%M%S')
    meta = dict(profile)
    meta.update({"version": version, "labels": labels, "files": files, "metrics": metrics,
                 "input": {"name": "holistic_sequence", "shape": [1, t, HOLISTIC_SIZE], "dtype": "float32"},
                 "output": {"name": "probabilities", "shape": [1, num_classes]}})
    with open(os.path.join(out_dir, 'profile.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Profile '{name}' done: val accuracy {val_acc:.3f}. Saved to {out_dir} (version {version}).")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train sign classifier for a quality profile.")
    parser.add_argument('--profile', default=None,
                        help="Profile name from config/profiles.json, or 'all'. Defaults to $PROFILE or the config default.")
    parser.add_argument('--data', default=DATA_PATH)
    parser.add_argument('--out', default=MODELS_PATH)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None, help="Override the profile's epochs")
    parser.add_argument('--batch-size', type=int, default=None, help="Override the profile's batch size")
    args = parser.parse_args()

    names = list(load_profiles()['profiles']) if args.profile == 'all' else [get_profile(args.profile)['name']]
    results = [train_profile(n, args.data, args.out, args.seed, args.epochs, args.batch_size) for n in names]
    if any(r is None for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
