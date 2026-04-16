# Roadmap: Sign Language to Text Translation

This document outlines the architectural path to evolve this project from a static letter recognizer to a real-time sign language translator.

## Phase 1: Landmark Extraction (COMPLETED)
*Status:* Switched from image saves to coordinate (.npy) saves. Verified via visualization tools.

## Phase 2: Temporal Sequence Modeling (ACTIVE)
*Status:* Created `train_model.py`. Now using an LSTM architecture to learn patterns across 30-frame sequences.

## Phase 3: Sentence Reconstruction (COMPLETED)
*Status:* Implemented a word debouncer, sentence buffer, and a Gloss-to-Text translator utility.

## Phase 4: Holistic Body/Face Tracking (ACTIVE)
*Status:* Researching MediaPipe Holistic integration to capture emotional context and full-body signs.


---

## Suggested Development Stack
- **Features:** MediaPipe (Holistic/Hands)
- **Deep Learning:** TensorFlow/Keras or PyTorch
- **NLP:** HuggingFace Transformers
- **Deployment:** [Streamlit](https://streamlit.io/) for a web interface or [TensorFlow Lite](https://www.tensorflow.org/lite) for mobile.
