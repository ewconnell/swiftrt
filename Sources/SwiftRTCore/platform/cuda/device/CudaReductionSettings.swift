//******************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import SwiftRTCuda

public final class ReductionSettings<S: TensorShape, E: StorageElement> {
    // properties
    public let reduction: ReductionTensorDescriptor
    public let xDesc, oDesc: TensorDescriptor
    public let workspaceSize: Int
    @usableFromInline var _workspace: DeviceMemory!

    @inlinable public init(
        op: ReductionOp,
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>,
        using queue: Platform.Device.Queue,
        nan: NanPropagation = .propagate
    ) {
        if S.rank >= 4 {
            xDesc = x.createTensorDescriptor()
            oDesc = out.createTensorDescriptor()
        } else {
            xDesc = Tensor<Shape4,E>(indenting: x).createTensorDescriptor()
            oDesc = Tensor<Shape4,E>(indenting: out).createTensorDescriptor()
        }
        
        reduction = ReductionTensorDescriptor(op: op, nan: nan, dataType: E.type)

        var size = 0
        cudaCheck(cudnnGetReductionWorkspaceSize(
            queue.cudnn.handle,
            reduction.desc,
            xDesc.desc,
            oDesc.desc,
            &size
        ))
        workspaceSize = size
    }

    @inlinable public func getWorkspace(
        using queue: Platform.Device.Queue
    ) -> DeviceMemory {
        guard _workspace == nil else { return _workspace }
        _workspace = queue.allocate(byteCount: workspaceSize)
        return _workspace
    }

    @inlinable public func releaseWorkspace() {
        _workspace = nil
    }
}