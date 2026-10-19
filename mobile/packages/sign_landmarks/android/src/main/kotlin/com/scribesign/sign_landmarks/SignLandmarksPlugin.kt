package com.scribesign.sign_landmarks

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min

/** MediaPipe HolisticLandmarker -> 1662-float frames (layout in lib/sign_landmarks.dart). */
class SignLandmarksPlugin : FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context
    private lateinit var executor: ExecutorService
    private val main = Handler(Looper.getMainLooper())
    private var landmarker: HolisticLandmarker? = null
    private var classifier: Interpreter? = null
    private var lastTimestamp = -1L

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        context = binding.applicationContext
        executor = Executors.newSingleThreadExecutor()
        channel = MethodChannel(binding.binaryMessenger, "sign_landmarks")
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        executor.execute { closeLandmarker(); closeClassifier() }
        executor.shutdown()
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        // MediaPipe calls are synchronous: run them off the UI thread, one at a time.
        executor.execute {
            try {
                val value: Any? = when (call.method) {
                    "initialize" -> initialize(
                        call.argument<String>("modelPath")!!,
                        call.argument<Boolean>("useGpu") ?: true,
                    )
                    "detect" -> detect(call)
                    "close" -> { closeLandmarker(); null }
                    "loadClassifier" -> loadClassifier(call.argument<String>("modelPath")!!)
                    "classify" -> classify(call.argument<FloatArray>("input")!!)
                    "closeClassifier" -> { closeClassifier(); null }
                    else -> { main.post { result.notImplemented() }; return@execute }
                }
                main.post { result.success(value) }
            } catch (e: Exception) {
                main.post { result.error("sign_landmarks", e.message ?: e.toString(), null) }
            }
        }
    }

    private fun initialize(modelPath: String, useGpu: Boolean): String {
        closeLandmarker()
        val bytes = File(modelPath).readBytes()
        val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        var lastError: Exception? = null
        for (delegate in if (useGpu) listOf(Delegate.GPU, Delegate.CPU) else listOf(Delegate.CPU)) {
            try {
                buffer.rewind()
                val options = HolisticLandmarker.HolisticLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(buffer).setDelegate(delegate).build())
                    .setRunningMode(RunningMode.VIDEO)
                    .setMinPoseDetectionConfidence(0.5f)
                    .setMinPosePresenceConfidence(0.5f)
                    .setMinFaceDetectionConfidence(0.5f)
                    .setMinHandLandmarksConfidence(0.5f)
                    .build()
                landmarker = HolisticLandmarker.createFromOptions(context, options)
                lastTimestamp = -1L
                return delegate.name
            } catch (e: Exception) {
                lastError = e
            }
        }
        throw lastError ?: IllegalStateException("Could not create HolisticLandmarker")
    }

    private fun closeLandmarker() {
        landmarker?.close()
        landmarker = null
    }

    private fun loadClassifier(modelPath: String): List<Int> {
        closeClassifier()
        val interpreter = Interpreter(File(modelPath), Interpreter.Options().setNumThreads(2))
        classifier = interpreter
        val input = interpreter.getInputTensor(0).shape()   // [1, frames, features]
        val output = interpreter.getOutputTensor(0).shape() // [1, classes]
        return listOf(input[1], input[2], output[1])
    }

    private fun classify(input: FloatArray): FloatArray {
        val interpreter = classifier ?: throw IllegalStateException("Call loadClassifier() first")
        val buffer = ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(input)
        val output = Array(1) { FloatArray(interpreter.getOutputTensor(0).shape()[1]) }
        interpreter.run(buffer, output)
        return output[0]
    }

    private fun closeClassifier() {
        classifier?.close()
        classifier = null
    }

    private fun detect(call: MethodCall): FloatArray {
        val lm = landmarker ?: throw IllegalStateException("Call initialize() first")
        val bytes = call.argument<ByteArray>("bytes")!!
        val raw = when (call.argument<String>("format")) {
            "jpeg" -> BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: throw IllegalArgumentException("Invalid JPEG frame")
            "nv21" -> nv21ToBitmap(bytes, call.argument<Int>("width")!!, call.argument<Int>("height")!!)
            else -> throw IllegalArgumentException("Unsupported format on Android: ${call.argument<String>("format")}")
        }
        val bitmap = transform(
            raw,
            call.argument<Int>("rotation") ?: 0,
            call.argument<Boolean>("mirror") ?: true,
            call.argument<Int>("targetWidth") ?: 0,
        )
        var ts = (call.argument<Number>("timestampMs") ?: 0).toLong()
        if (ts <= lastTimestamp) ts = lastTimestamp + 1
        lastTimestamp = ts
        val result = lm.detectForVideo(BitmapImageBuilder(bitmap).build(), ts)
        return toFrame(result)
    }

    /** Rotate clockwise to upright, mirror horizontally (selfie view), downscale to targetWidth. */
    private fun transform(src: Bitmap, rotation: Int, mirror: Boolean, targetWidth: Int): Bitmap {
        val uprightWidth = if (rotation % 180 == 0) src.width else src.height
        val scale = if (targetWidth in 1 until uprightWidth) targetWidth.toFloat() / uprightWidth else 1f
        if (rotation % 360 == 0 && !mirror && scale == 1f) return src
        val m = Matrix()
        if (rotation % 360 != 0) m.postRotate(rotation.toFloat())
        if (mirror) m.postScale(-1f, 1f)
        if (scale != 1f) m.postScale(scale, scale)
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun nv21ToBitmap(nv21: ByteArray, width: Int, height: Int): Bitmap {
        val argb = IntArray(width * height)
        val frameSize = width * height
        for (j in 0 until height) {
            var uvp = frameSize + (j shr 1) * width
            var u = 0
            var v = 0
            for (i in 0 until width) {
                val y = maxOf(0, (nv21[j * width + i].toInt() and 0xff) - 16)
                if (i and 1 == 0) {
                    v = (nv21[uvp++].toInt() and 0xff) - 128
                    u = (nv21[uvp++].toInt() and 0xff) - 128
                }
                val y1192 = 1192 * y
                val r = (y1192 + 1634 * v).coerceIn(0, 262143)
                val g = (y1192 - 833 * v - 400 * u).coerceIn(0, 262143)
                val b = (y1192 + 2066 * u).coerceIn(0, 262143)
                argb[j * width + i] = -0x1000000 or ((r shl 6) and 0xff0000) or ((g shr 2) and 0xff00) or ((b shr 10) and 0xff)
            }
        }
        return Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun toFrame(r: HolisticLandmarkerResult): FloatArray {
        val out = FloatArray(FRAME_SIZE)
        r.poseLandmarks().take(33).forEachIndexed { i, p ->
            out[i * 4] = p.x(); out[i * 4 + 1] = p.y(); out[i * 4 + 2] = p.z()
            out[i * 4 + 3] = p.visibility().orElse(0f)
        }
        put(out, r.faceLandmarks(), FACE_OFFSET, 468)  // Tasks face model has 478 points; keep the 468 mesh
        put(out, r.leftHandLandmarks(), LEFT_HAND_OFFSET, 21)
        put(out, r.rightHandLandmarks(), RIGHT_HAND_OFFSET, 21)
        return out
    }

    private fun put(out: FloatArray, points: List<NormalizedLandmark>, offset: Int, count: Int) {
        for (i in 0 until min(count, points.size)) {
            val p = points[i]
            out[offset + i * 3] = p.x(); out[offset + i * 3 + 1] = p.y(); out[offset + i * 3 + 2] = p.z()
        }
    }

    companion object {
        const val FRAME_SIZE = 1662
        const val FACE_OFFSET = 132
        const val LEFT_HAND_OFFSET = 1536
        const val RIGHT_HAND_OFFSET = 1599
    }
}
