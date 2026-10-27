// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "custom_op_local_function.h"

#include <cmath>
#include <mutex>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"
#include "custom_mul_add.h"

static const char* c_OpDomain = "onnx_extented.ortops.tutorial.cpu";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                          const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  Ort::UnownedSessionOptions session_options(options);

  // An instance remaining available until onnxruntime unload the library.
  static Cpu::CustomMulAddOp c_CustomMulAddFloat(
      "CustomMulAdd", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      false);
  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_CustomMulAddFloat);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }

  return result;
}
