#include <limits.h>
#include <atomic>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

REGISTER_OP("MaxBytesInUse").Output("out: int64").SetIsStateful();

// Op that measures the peak memory in bytes.
class MaxBytesInUseOp : public OpKernel {
 public:
  explicit MaxBytesInUseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Allocator* allocator =
        context->device()->GetAllocator(AllocatorAttributes());
    AllocatorStats allocator_stats;
    allocator->GetStats(&allocator_stats);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<int64>()() = allocator_stats.max_bytes_in_use;
  }
};

// MallocExtension_GetAllocatedSize doesn't return the allocated size reliably
// for CPU allocators, so we register this op on GPU only.
REGISTER_KERNEL_BUILDER(Name("MaxBytesInUse").Device(DEVICE_GPU).HostMemory("out"), MaxBytesInUseOp);
