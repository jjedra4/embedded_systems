// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <vector>

namespace Cpu {

struct CustomMulAddKernel {
  CustomMulAddKernel(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
};

struct CustomMulAddOp : Ort::CustomOpBase<CustomMulAddOp, CustomMulAddKernel> {
  typedef Ort::CustomOpBase<CustomMulAddOp, CustomMulAddKernel> parent_type;
  CustomMulAddOp(const char* op_name, ONNXTensorElementDataType ab_type,
               ONNXTensorElementDataType c_type,
               ONNXTensorElementDataType d_type, bool compute_time_as_output)
      : parent_type() {
    op_name_ = op_name;
    ab_type_ = ab_type;
    c_type_ = c_type;
    d_type_ = d_type;
    compute_time_as_output_ = compute_time_as_output;
  }
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  const char* GetExecutionProviderType() const;

  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const;

  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const;

 private:
  const char* op_name_;
  ONNXTensorElementDataType ab_type_;
  ONNXTensorElementDataType c_type_;
  ONNXTensorElementDataType d_type_;
  bool compute_time_as_output_;
};

}  // namespace Cpu
