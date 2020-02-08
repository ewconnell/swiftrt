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

//==============================================================================
// setGlobal(platform:
// NOTE: do this in your app if the source is part of the app

// let globalPlatform = Platform<XyzService>()
// You can define APPCOLOCATED in the build, or just delete this file

//==============================================================================
/// setGlobal(platform:
/// this is used to set the framework global variable for static functions
/// and free floating objects to access the platform
#if !APPCOLOCATED
@inlinable
public func setGlobal<T>(platform: T) -> T where T: ComputePlatform {
    globalPlatform = platform
    return platform
}

/// This is an existential, so it is slower than if the
public var globalPlatform: PlatformFunctions = Platform<CpuService>()
#endif
