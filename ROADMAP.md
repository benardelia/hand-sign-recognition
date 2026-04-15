# Roadmap: Sign Language to Text Translation

This document outlines the architectural path to evolve this project from a static letter recognizer to a real-time sign language translator.

## Phase 1: Landmark Extraction (COMPLETED)
*Status:* Switched from image saves to coordinate (.npy) saves. Verified via visualization tools.

## Phase 2: Temporal Sequence Modeling (ACTIVE)
*Status:* Created `train_model.py`. Now using an LSTM architecture to learn patterns across 30-frame sequences.

## Phase 3: Sentence Reconstruction (NLP)
*Current State:* Predicts one label at a time.
*Limitation:* Sign language grammar (Glosses) differs from spoken English.

**The Move:**
- Integrate a **Gloss-to-Text** transformer (e.g., T5 or a lightweight LLM).
- Map a sequence of detected signs (e.g., `[STORE, GO, I]`) to a natural sentence (`"I am going to the store."`).
- **Goal:** Produce human-readable, grammatically correct translations.

## Phase 4: Holistic Tracking (Contextual Signs)
*Current State:* Tracks hands only.
*Limitation:* Some signs depend on facial expressions or body posture.

**The Move:**
- Upgrade to `MediaPipe Holistic`.
- Incorporate **Face Mesh** and **Pose** landmarks into the training data.
- **Goal:** Improve accuracy for advanced signs that rely on non-manual markers.

---

## Suggested Development Stack
- **Features:** MediaPipe (Holistic/Hands)
- **Deep Learning:** TensorFlow/Keras or PyTorch
- **NLP:** HuggingFace Transformers
- **Deployment:** [Streamlit](https://streamlit.io/) for a web interface or [TensorFlow Lite](https://www.tensorflow.org/lite) for mobile.
