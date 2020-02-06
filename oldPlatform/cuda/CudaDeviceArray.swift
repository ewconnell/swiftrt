//******************************************************************************
// Copyright 2019 Google LLC
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
import Foundation
import CCuda

public class CudaDeviceArray : DeviceArray {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public let buffer: UnsafeMutableRawBufferPointer
    public var device: ComputeDevice { return cudaDevice }
    public var version = 0
    public let isReadOnly: Bool
    private let cudaDevice: CudaDevice

    //--------------------------------------------------------------------------
    // initializers
    public init(device: CudaDevice, byteCount: Int, zero: Bool) throws {
        cudaDevice = device
        self.isReadOnly = false
        try cudaDevice.select()
		var pointer: UnsafeMutableRawPointer?
		try cudaCheck(status: cudaMalloc(&pointer, byteCount))
        if zero {
            try cudaCheck(status: cudaMemset(&pointer, 0, byteCount))
        }
        buffer = UnsafeMutableRawBufferPointer(start: pointer, count: byteCount)
		trackingId = ObjectTracker.global.register(
                self, supplementalInfo: "count: \(byteCount)")
	}

    //--------------------------------------------------------------------------
    /// readOnly uma buffer
    public init(device: CudaDevice, buffer: UnsafeRawBufferPointer) {
        assert(buffer.baseAddress != nil)
        cudaDevice = device
        self.isReadOnly = true
        let pointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        self.buffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                    count: buffer.count)
        self.trackingId = ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    /// readWrite uma buffer
    public init(device: CudaDevice, buffer: UnsafeMutableRawBufferPointer) {
        cudaDevice = device
        self.isReadOnly = false
        self.buffer = buffer
        self.trackingId = ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    // cleanup
    deinit {
		do {
			try cudaDevice.select()
			try cudaCheck(status: cudaFree(buffer.baseAddress!))
			ObjectTracker.global.remove(trackingId: trackingId)
		} catch {
            cudaDevice.writeLog(
                    "\(releaseString) CudaDeviceArray(\(trackingId)) " +
                    "\(String(describing: error))")
		}
	}
}
