// Subset of the TensorFlow Lite C API (tensorflow/lite/core/c/c_api.h).
//
// MediaPipeTasksCommon force-loads a graph library that already contains the
// TFLite runtime, so linking a second TFLite (e.g. TensorFlowLiteC) fails with
// duplicate symbols. The sign classifier calls these functions instead, which
// resolve to MediaPipe's copy of the runtime.
#ifndef SIGN_LANDMARKS_TFLITE_H_
#define SIGN_LANDMARKS_TFLITE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteTensor TfLiteTensor;

/// TfLiteStatus; 0 == kTfLiteOk.
typedef int SignLandmarksTfLiteStatus;

TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);
void TfLiteModelDelete(TfLiteModel* model);

TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate(void);
void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options, int32_t num_threads);
void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options);

TfLiteInterpreter* TfLiteInterpreterCreate(const TfLiteModel* model,
                                           const TfLiteInterpreterOptions* optional_options);
void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);
SignLandmarksTfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter* interpreter);
SignLandmarksTfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter* interpreter);
TfLiteTensor* TfLiteInterpreterGetInputTensor(const TfLiteInterpreter* interpreter, int32_t input_index);
const TfLiteTensor* TfLiteInterpreterGetOutputTensor(const TfLiteInterpreter* interpreter,
                                                     int32_t output_index);

int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);
int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t dim_index);
size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);
SignLandmarksTfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor, const void* input_data,
                                                    size_t input_data_size);
SignLandmarksTfLiteStatus TfLiteTensorCopyToBuffer(const TfLiteTensor* output_tensor, void* output_data,
                                                  size_t output_data_size);

#ifdef __cplusplus
}
#endif

#endif  // SIGN_LANDMARKS_TFLITE_H_
