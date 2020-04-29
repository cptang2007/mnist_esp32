/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "esp_log.h"
#include "mnist_model_data.h"
#include "number_data.h"

static const char* TAGD = "Debug";
static const char* TAGI = "Inference";

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

uint32_t inference_count = 0;
float Prob;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 80 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

const float* test_number;

const char keyword[10][2] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};	//declare a C++ String

uint32_t startTime;

// The name of this function is important for Arduino compatibility.
void setup() {

  ESP_LOGD(TAGD, "Initialize...");
  startTime = esp_log_timestamp();
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.

  ESP_LOGD(TAGD, "Loading Model...");


  model = tflite::GetModel(mnist_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  //static tflite::ops::micro::AllOpsResolver resolver;

  static tflite::MicroMutableOpResolver resolver;
  resolver.AddBuiltin(
		  tflite::BuiltinOperator_QUANTIZE,
          tflite::ops::micro::Register_QUANTIZE()
  );
  resolver.AddBuiltin(
		  tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
		  tflite::ops::micro::Register_DEPTHWISE_CONV_2D(),
		  1, 3
  );
  resolver.AddBuiltin(
  		  tflite::BuiltinOperator_MAX_POOL_2D,
  		  tflite::ops::micro::Register_MAX_POOL_2D(),
  		  1, 2
  );
  resolver.AddBuiltin(
  		  tflite::BuiltinOperator_CONV_2D,
  		  tflite::ops::micro::Register_CONV_2D(),
  		  1, 3
  );
  resolver.AddBuiltin(
		  tflite::BuiltinOperator_FULLY_CONNECTED,
          tflite::ops::micro::Register_FULLY_CONNECTED(),
		  1, 4
  );
  resolver.AddBuiltin(
		  tflite::BuiltinOperator_SOFTMAX,
		  tflite::ops::micro::Register_SOFTMAX(),
		  1, 2
  );
  resolver.AddBuiltin(
  		  tflite::BuiltinOperator_DEQUANTIZE,
          tflite::ops::micro::Register_DEQUANTIZE(),
		  1, 2
  );

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  ESP_LOGD(TAGD, "Initialize DONE (%dms)", esp_log_timestamp()-startTime);
}

// The name of this function is important for Arduino compatibility.
void inference() {

	test_number = number_2_data;

	ESP_LOGD(TAGI, "Loading number data...");
	for (int i = 0; i < 784; i++) {
		input-> data.f[i] = test_number[i];
	}

	ESP_LOGD(TAGD, "Inference Start...");
	startTime = esp_log_timestamp();

	// Run the model on the spectrogram input and make sure it succeeds.
	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
		return ;
	}

	//TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

	for (int i=0; i <10; i++) {
		Prob = output-> data.f[i];
		ESP_LOGI(TAGI, "Prob. of '%s'\t: %f", keyword[i], Prob);
	}

	ESP_LOGD(TAGD, "Inference End (%dms)", esp_log_timestamp()-startTime);

	return;
}
