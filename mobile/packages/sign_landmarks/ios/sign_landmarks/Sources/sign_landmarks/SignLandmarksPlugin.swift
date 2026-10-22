import CoreImage
import Flutter
import MediaPipeTasksVision
import UIKit

/// MediaPipe HolisticLandmarker -> 1662-float frames (layout in lib/sign_landmarks.dart).
public class SignLandmarksPlugin: NSObject, FlutterPlugin {
  private static let frameSize = 1662
  private static let faceOffset = 132
  private static let leftHandOffset = 1536
  private static let rightHandOffset = 1599

  // MediaPipe calls are synchronous: run them off the main thread, one at a time.
  private let queue = DispatchQueue(label: "com.scribesign.sign_landmarks")
  private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
  private var landmarker: HolisticLandmarker?
  private var lastTimestamp = -1
  private var classifier: TFLiteClassifier?

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "sign_landmarks", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(SignLandmarksPlugin(), channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let args = call.arguments as? [String: Any] ?? [:]
    queue.async {
      let reply: Any?
      do {
        switch call.method {
        case "initialize":
          reply = try self.initialize(
            modelPath: args["modelPath"] as? String ?? "",
            useGpu: args["useGpu"] as? Bool ?? true)
        case "detect":
          reply = try self.detect(args)
        case "close":
          self.landmarker = nil
          reply = nil
        case "loadClassifier":
          let model = try TFLiteClassifier(modelPath: args["modelPath"] as? String ?? "")
          self.classifier = model
          reply = [model.frames, model.features, model.classes]
        case "classify":
          guard let classifier = self.classifier else { throw PluginError.message("Call loadClassifier() first") }
          guard let input = (args["input"] as? FlutterStandardTypedData)?.data else {
            throw PluginError.message("Missing classifier input")
          }
          reply = FlutterStandardTypedData(float32: try classifier.run(input))
        case "closeClassifier":
          self.classifier = nil
          reply = nil
        default:
          DispatchQueue.main.async { result(FlutterMethodNotImplemented) }
          return
        }
      } catch {
        DispatchQueue.main.async {
          result(FlutterError(code: "sign_landmarks", message: error.localizedDescription, details: nil))
        }
        return
      }
      DispatchQueue.main.async { result(reply) }
    }
  }

  private func initialize(modelPath: String, useGpu: Bool) throws -> String {
    landmarker = nil
    var lastError: Error?
    for delegate in useGpu ? [Delegate.GPU, Delegate.CPU] : [Delegate.CPU] {
      do {
        let options = HolisticLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.baseOptions.delegate = delegate
        options.runningMode = .video
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minFaceDetectionConfidence = 0.5
        options.minHandLandmarksConfidence = 0.5
        landmarker = try HolisticLandmarker(options: options)
        lastTimestamp = -1
        return delegate == .GPU ? "GPU" : "CPU"
      } catch {
        lastError = error
      }
    }
    throw lastError ?? PluginError.message("Could not create HolisticLandmarker")
  }

  private func detect(_ args: [String: Any]) throws -> FlutterStandardTypedData {
    guard let landmarker = landmarker else { throw PluginError.message("Call initialize() first") }
    guard let data = (args["bytes"] as? FlutterStandardTypedData)?.data else {
      throw PluginError.message("Missing frame bytes")
    }
    let format = args["format"] as? String ?? "jpeg"
    var image: CIImage
    switch format {
    case "jpeg":
      guard let decoded = CIImage(data: data) else { throw PluginError.message("Invalid JPEG frame") }
      image = decoded
    case "bgra8888":
      let width = args["width"] as? Int ?? 0
      let height = args["height"] as? Int ?? 0
      let bytesPerRow = args["bytesPerRow"] as? Int ?? width * 4
      image = CIImage(bitmapData: data, bytesPerRow: bytesPerRow,
                      size: CGSize(width: width, height: height), format: .BGRA8,
                      colorSpace: CGColorSpaceCreateDeviceRGB())
    default:
      throw PluginError.message("Unsupported format on iOS: \(format)")
    }

    image = transform(image, rotation: args["rotation"] as? Int ?? 0,
                      mirror: args["mirror"] as? Bool ?? true, targetWidth: args["targetWidth"] as? Int ?? 0)
    guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
      throw PluginError.message("Could not render frame")
    }
    let mpImage = try MPImage(uiImage: UIImage(cgImage: cgImage))

    var ts = args["timestampMs"] as? Int ?? 0
    if ts <= lastTimestamp { ts = lastTimestamp + 1 }
    lastTimestamp = ts
    let result = try landmarker.detect(videoFrame: mpImage, timestampInMilliseconds: ts)
    return FlutterStandardTypedData(float32: toFrame(result))
  }

  /// Rotate clockwise to upright, mirror horizontally (selfie view), downscale to targetWidth.
  private func transform(_ input: CIImage, rotation: Int, mirror: Bool, targetWidth: Int) -> CIImage {
    var image = input
    switch ((rotation % 360) + 360) % 360 {
    case 90: image = image.oriented(.right)
    case 180: image = image.oriented(.down)
    case 270: image = image.oriented(.left)
    default: break
    }
    if mirror {
      image = image.oriented(.upMirrored)
    }
    let width = image.extent.width
    if targetWidth > 0 && CGFloat(targetWidth) < width {
      let scale = CGFloat(targetWidth) / width
      image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }
    // Move the extent back to the origin after the transforms.
    return image.transformed(by: CGAffineTransform(translationX: -image.extent.origin.x,
                                                   y: -image.extent.origin.y))
  }

  private func toFrame(_ r: HolisticLandmarkerResult) -> Data {
    var out = [Float](repeating: 0, count: SignLandmarksPlugin.frameSize)
    for (i, p) in r.poseLandmarks.prefix(33).enumerated() {
      out[i * 4] = p.x
      out[i * 4 + 1] = p.y
      out[i * 4 + 2] = p.z
      out[i * 4 + 3] = p.visibility?.floatValue ?? 0
    }
    // The Tasks face model returns 478 points (468 mesh + 10 iris); keep the 468 mesh points.
    put(&out, r.faceLandmarks, offset: SignLandmarksPlugin.faceOffset, count: 468)
    put(&out, r.leftHandLandmarks, offset: SignLandmarksPlugin.leftHandOffset, count: 21)
    put(&out, r.rightHandLandmarks, offset: SignLandmarksPlugin.rightHandOffset, count: 21)
    return out.withUnsafeBufferPointer { Data(buffer: $0) }
  }

  private func put(_ out: inout [Float], _ points: [NormalizedLandmark], offset: Int, count: Int) {
    for (i, p) in points.prefix(count).enumerated() {
      out[offset + i * 3] = p.x
      out[offset + i * 3 + 1] = p.y
      out[offset + i * 3 + 2] = p.z
    }
  }
}

/// Sign classifier on the TFLite C API that MediaPipe already links (see SignLandmarksTFLite.h).
final class TFLiteClassifier {
  let frames: Int
  let features: Int
  let classes: Int
  private let model: OpaquePointer
  private let options: OpaquePointer
  private let interpreter: OpaquePointer

  init(modelPath: String) throws {
    guard let model = TfLiteModelCreateFromFile(modelPath) else {
      throw PluginError.message("Cannot load classifier model at \(modelPath)")
    }
    guard let options = TfLiteInterpreterOptionsCreate() else {
      TfLiteModelDelete(model)
      throw PluginError.message("Cannot create interpreter options")
    }
    TfLiteInterpreterOptionsSetNumThreads(options, 2)
    guard let interpreter = TfLiteInterpreterCreate(model, options) else {
      TfLiteInterpreterOptionsDelete(options)
      TfLiteModelDelete(model)
      throw PluginError.message("Cannot create the classifier interpreter")
    }
    let input = TfLiteInterpreterGetInputTensor(interpreter, 0)
    let output = TfLiteInterpreterGetOutputTensor(interpreter, 0)
    guard TfLiteInterpreterAllocateTensors(interpreter) == 0,
          TfLiteTensorNumDims(input) == 3, TfLiteTensorNumDims(output) == 2 else {
      TfLiteInterpreterDelete(interpreter)
      TfLiteInterpreterOptionsDelete(options)
      TfLiteModelDelete(model)
      throw PluginError.message("Unexpected classifier model (expected input [1, T, F], output [1, C])")
    }
    self.model = model
    self.options = options
    self.interpreter = interpreter
    frames = Int(TfLiteTensorDim(input, 1))
    features = Int(TfLiteTensorDim(input, 2))
    classes = Int(TfLiteTensorDim(output, 1))
  }

  deinit {
    TfLiteInterpreterDelete(interpreter)
    TfLiteInterpreterOptionsDelete(options)
    TfLiteModelDelete(model)
  }

  func run(_ input: Data) throws -> Data {
    let inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0)
    guard input.count == TfLiteTensorByteSize(inputTensor) else {
      throw PluginError.message("Classifier input has \(input.count) bytes, expected \(TfLiteTensorByteSize(inputTensor))")
    }
    let copied = input.withUnsafeBytes { TfLiteTensorCopyFromBuffer(inputTensor, $0.baseAddress, input.count) }
    guard copied == 0, TfLiteInterpreterInvoke(interpreter) == 0 else {
      throw PluginError.message("Classifier inference failed")
    }
    let outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0)
    var output = Data(count: TfLiteTensorByteSize(outputTensor))
    let size = output.count
    let status = output.withUnsafeMutableBytes { TfLiteTensorCopyToBuffer(outputTensor, $0.baseAddress, size) }
    guard status == 0 else { throw PluginError.message("Cannot read classifier output") }
    return output
  }
}

enum PluginError: LocalizedError {
  case message(String)

  var errorDescription: String? {
    if case .message(let text) = self { return text }
    return nil
  }
}
