// Browser-side holistic landmark extraction with MediaPipe Tasks (HolisticLandmarker).
//
// Produces the same 1662-float frame as utils/holistic_utils.py:
//   pose 33 x (x, y, z, visibility) | face 468 x (x, y, z) | left hand 21 x 3 | right hand 21 x 3
// Missing parts are zeros. The camera image is mirrored (selfie view) before
// detection, matching the Python collectors. Only numbers leave the browser.

export const HOLISTIC_SIZE = 1662;
const FACE_OFFSET = 132, LH_OFFSET = 1536, RH_OFFSET = 1599;

export function resultToFrame(res) {
    const out = new Float32Array(HOLISTIC_SIZE);
    const pose = res.poseLandmarks && res.poseLandmarks[0];
    if (pose) {
        for (let i = 0; i < 33 && i < pose.length; i++) {
            const p = pose[i];
            out[i * 4] = p.x; out[i * 4 + 1] = p.y; out[i * 4 + 2] = p.z;
            out[i * 4 + 3] = p.visibility ?? 0;
        }
    }
    const put = (points, offset, count) => {
        if (!points) return;
        // The Tasks face model returns 478 points (468 mesh + 10 iris); keep the 468 mesh points.
        for (let i = 0; i < count && i < points.length; i++) {
            const p = points[i];
            out[offset + i * 3] = p.x; out[offset + i * 3 + 1] = p.y; out[offset + i * 3 + 2] = p.z;
        }
    };
    put(res.faceLandmarks && res.faceLandmarks[0], FACE_OFFSET, 468);
    put(res.leftHandLandmarks && res.leftHandLandmarks[0], LH_OFFSET, 21);
    put(res.rightHandLandmarks && res.rightHandLandmarks[0], RH_OFFSET, 21);
    return out;
}

export class HolisticPipeline {
    constructor(mediapipeConfig) {
        this.cfg = mediapipeConfig;
        this.video = document.createElement('video');
        this.video.playsInline = true;
        this.video.muted = true;
        this.work = document.createElement('canvas');      // mirrored, profile-resolution frame fed to MediaPipe
        this.canvas = document.createElement('canvas');    // what the user sees
        this.canvas.className = 'feed-img live-canvas';
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.running = false;
        this.camera = { width: 640, height: 480, fps: 30 };
        this.lastTs = 0;
        this.lastRun = 0;
        this.fpsEstimate = 0;
        this.onFrame = null;
        this.onOverlay = null;
        this.onError = null;
        this.failures = 0;
    }

    async load() {
        const vision = await import(this.cfg.bundle);
        this.vision = vision;
        const fileset = await vision.FilesetResolver.forVisionTasks(this.cfg.wasm);
        const options = (delegate) => ({
            baseOptions: { modelAssetPath: this.cfg.model, delegate },
            runningMode: 'VIDEO',
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minHandLandmarksConfidence: 0.5,
            minFaceDetectionConfidence: 0.5,
        });
        try {
            this.landmarker = await vision.HolisticLandmarker.createFromOptions(fileset, options('GPU'));
            this.delegate = 'GPU';
        } catch (e) {
            console.warn('GPU delegate unavailable, using CPU:', e);
            this.landmarker = await vision.HolisticLandmarker.createFromOptions(fileset, options('CPU'));
            this.delegate = 'CPU';
        }
        this.drawing = new vision.DrawingUtils(this.ctx);
    }

    setCamera(camera) {
        this.camera = { ...camera };
    }

    async startCamera(deviceId) {
        this.stopCamera();
        const video = {
            width: { ideal: this.camera.width },
            height: { ideal: this.camera.height },
            frameRate: { ideal: this.camera.fps },
        };
        if (deviceId) video.deviceId = { exact: deviceId };
        else video.facingMode = 'user';
        this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
        this.video.srcObject = this.stream;
        await this.video.play();
    }

    stopCamera() {
        if (this.stream) this.stream.getTracks().forEach(t => t.stop());
        this.stream = null;
    }

    start() {
        if (this.running) return;
        this.running = true;
        const tick = () => {
            if (!this.running) return;
            this._step();
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }

    stop() {
        this.running = false;
    }

    _step() {
        const v = this.video;
        if (!this.landmarker || v.readyState < 2 || !v.videoWidth) return;
        const now = performance.now();
        if (now - this.lastRun < 1000 / this.camera.fps - 2) return;   // pace to the profile fps
        if (this.lastRun) this.fpsEstimate = 0.9 * this.fpsEstimate + 0.1 * (1000 / (now - this.lastRun));
        this.lastRun = now;

        // Process at the profile resolution (keeps the camera's aspect ratio).
        const w = Math.min(this.camera.width, v.videoWidth);
        const h = Math.round(w * v.videoHeight / v.videoWidth);
        if (this.work.width !== w || this.work.height !== h) {
            this.work.width = w; this.work.height = h;
            this.canvas.width = w; this.canvas.height = h;
        }
        const wctx = this.work.getContext('2d');
        wctx.save();
        wctx.translate(w, 0);
        wctx.scale(-1, 1);
        wctx.drawImage(v, 0, 0, w, h);
        wctx.restore();

        const ts = Math.max(Math.round(now), this.lastTs + 1);
        this.lastTs = ts;
        let result;
        try {
            result = this.landmarker.detectForVideo(this.work, ts);
        } catch (e) {
            this.failures++;
            if (this.failures <= 3) console.error('Holistic detection failed:', e);
            // Usually means WebGL is unavailable (MediaPipe needs it even on the CPU delegate).
            if (this.failures === 30 && this.onError) this.onError(e);
            return;
        }
        this.failures = 0;
        const frame = resultToFrame(result);

        this.ctx.drawImage(this.work, 0, 0);
        this._draw(result);
        if (this.onFrame) this.onFrame(frame, ts, result);
        if (this.onOverlay) this.onOverlay(this.ctx, w, h);
    }

    _draw(res) {
        const H = this.vision.HolisticLandmarker;
        const d = this.drawing;
        const face = res.faceLandmarks && res.faceLandmarks[0];
        if (face) {
            for (const conns of [H.FACE_LANDMARKS_LIPS, H.FACE_LANDMARKS_LEFT_EYEBROW, H.FACE_LANDMARKS_RIGHT_EYEBROW,
                                 H.FACE_LANDMARKS_FACE_OVAL]) {
                if (conns) d.drawConnectors(face, conns, { color: 'rgba(80,255,121,0.6)', lineWidth: 1 });
            }
        }
        const pose = res.poseLandmarks && res.poseLandmarks[0];
        if (pose) d.drawConnectors(pose, H.POSE_CONNECTIONS, { color: 'rgba(99,102,241,0.9)', lineWidth: 2 });
        const hands = [[res.leftHandLandmarks, '#06b6d4'], [res.rightHandLandmarks, '#10b981']];
        for (const [list, color] of hands) {
            const hand = list && list[0];
            if (!hand) continue;
            d.drawConnectors(hand, H.HAND_CONNECTIONS, { color, lineWidth: 2 });
            d.drawLandmarks(hand, { color, radius: 2 });
        }
    }
}

window.HandsignPipeline = HolisticPipeline;
window.dispatchEvent(new Event('handsign-pipeline-ready'));
