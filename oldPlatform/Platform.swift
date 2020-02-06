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

//==============================================================================
// Platform
/// The root object to select compute services and devices
public final class Platform: LocalPlatform {
    // properties
    public var _defaultDevice: ComputeDevice?
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _errorMutex: Mutex = Mutex()
    public var _lastError: Error? = nil
    public var deviceIdPriority: [Int] = [0]
    public var id: Int = 0
    public static let local = Platform()
    public var serviceModuleDirectory = URL(fileURLWithPath: "TODO")
    public var servicePriority: [String] = []
    public lazy var services: [String : ComputeService] = {
        loadServices()
        return Platform._services!
    }()
    public static var _services: [String: ComputeService]?
    public private(set) var trackingId = 0
    public var logInfo: LogInfo
    
    /// This is used to provide a platform wide unique value for the
    /// `ComputeDevice.deviceArrayReplicaKey`
    private static var deviceArrayReplicaKeyCounter = AtomicCounter(value: -1)
    public static var nextDeviceArrayReplicaKey: Int {
        Platform.deviceArrayReplicaKeyCounter.increment()
    }

    //--------------------------------------------------------------------------
    // shortcut to the first unified memory cpu device
    public static var cpu: ComputeDevice = {
        for service in Platform.local.services.values {
            if service is CpuServiceProtocol &&
                service.devices[0].memory.addressing == .unified {
                return service.devices[0]
            }
        }
        fatalError("Missing CPU device. Must be enabled during build")
    }()

    public static var synchronousCpu: [ComputeDevice] = {
        Platform.local.services[cpuSynchronousServiceName]!.devices
    }()

    //--------------------------------------------------------------------------
    // these are to aid unit tests
    public static var testCpu1: ComputeDevice = {
        Platform.local.services[testCpuServiceName]!.devices[0]
    }()

    public static var testCpu2: ComputeDevice = {
        Platform.local.services[testCpuServiceName]!.devices[1]
    }()

    //--------------------------------------------------------------------------
    /// logWriter
    /// the caller can specify a root logWriter which will be inherited by the
    /// device queue hierarchy, but can be overridden at any point down
    /// the tree
    public var logWriter: Log {
        get { logInfo.logWriter }
        set { logInfo.logWriter = newValue }
    }
    
    //--------------------------------------------------------------------------
    // initializers
    /// `init` is private because this is a singleton. Use the `local` static
    /// member to access the shared instance.
    private init() {
        // create the log
        logInfo = LogInfo(logWriter: Log(isStatic: true), logLevel: .error,
                          namePath: String(describing: Platform.self),
                          nestingLevel: 0)
    }
}

//==============================================================================
/// a set of predefined property names to simplify configuring
/// the service properties
public enum CpuPropertyKey: Int {
    case queuesPerDevice
}

//==============================================================================
/// LocalPlatform
/// The default ComputePlatform implementation for a local host
public protocol LocalPlatform : ComputePlatform {
    /// the global services collection
    static var _services: [String: ComputeService]? { get set }
    var _defaultDevice: ComputeDevice? { get set }
    /// a platform wide unique key used during device initialization
    static var nextDeviceArrayReplicaKey: Int { get }
}

public extension LocalPlatform {
    //--------------------------------------------------------------------------
    /// log
    /// the caller can specify a root logWriter which will be inherited by the
    /// device queue hierarchy, but can be overridden at any point down
    /// the tree
    static var log: Log {
        get { Platform.local.logWriter }
        set { Platform.local.logWriter = newValue }
    }
    
    static var cpuConfiguration: [CpuPropertyKey: Any] { [
        .queuesPerDevice: 1
        ] }
    
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    /// The default platform error handler has nowhere else to go, so
    /// print the message, break to the debugger if possible, and exit.
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            print("Unhandled platform error: \(String(describing: error))")
//            raise(SIGINT)
        }
    }

    //--------------------------------------------------------------------------
    // loadServices
    // dynamically loads ComputeService bundles/dylib from the
    // `serviceModuleDirectory` and adds them to the `services` list
    func loadServices() {
        guard Platform._services == nil else { return }
        
        var loadedServices = [String: ComputeService]()
        do {
            //-------------------------------------
            // add synchronous cpu components
            #if canImport(CpuSync)
            loadedServices[cpuSynchronousServiceName] =
                try CpuSynchronousService(platform: Platform.local,
                                          id: loadedServices.count,
                                          logInfo: logInfo,
                                          name: cpuSynchronousServiceName)
            #endif
            
            //-------------------------------------
            // add asynchronous cpu components
            #if canImport(CpuAsync)
            loadedServices[cpuAsynchronousServiceName] =
                try CpuAsynchronousService(platform: Platform.local,
                                           id: loadedServices.count,
                                           logInfo: logInfo,
                                           name: cpuAsynchronousServiceName)
            #endif
            
            //-------------------------------------
            // add discreet test cpu service
            #if canImport(CpuTest)
            loadedServices[testCpuServiceName] =
                try TestCpuService(platform: Platform.local,
                                   id: loadedServices.count,
                                   logInfo: logInfo,
                                   name: testCpuServiceName)
            #endif
            
            //-------------------------------------
            // statically include driver modules if they are available
            #if canImport(CVulkan)
            loadedServices[vulkanServiceName] =
                try VulkanService(platform: Platform.local,
                                  id: loadedServices.count,
                                  logInfo: logInfo,
                                  name: vulkanServiceName)
            #endif

            #if canImport(CCuda)
            loadedServices[cudaServiceName] =
                try CudaService(platform: Platform.local,
                                id: loadedServices.count,
                                logInfo: logInfo,
                                name: cudaServiceName)
            #endif
            
            //-------------------------------------
            // dynamically load installed services
            let bundles = getPlugInBundles()
            for bundle in bundles {
                try bundle.loadAndReturnError()
                //            var unloadBundle = false
                
                if let serviceType =
                    bundle.principalClass as? ComputeService.Type {
                    
                    // create the service
                    let service =
                        try serviceType.init(platform: Platform.local,
                                             id: loadedServices.count,
                                             logInfo: logInfo, name: nil)
                    
                    diagnostic(
                        "Loaded compute service '\(service.name)'." +
                        " ComputeDevice count = \(service.devices.count)",
                        categories: .initialize)
                    
                    if service.devices.count > 0 {
                        // add plugin service
                        loadedServices[service.name] = service
                    } else {
                        writeLog("Compute service '\(service.name)' " +
                            "successfully loaded, but reported devices = 0, " +
                            "so service is unavailable", level: .warning)
                        //                    unloadBundle = true
                    }
                }
                // TODO: we should call bundle unload here if there were no devices
                // however simply calling bundle.load() then bundle.unload() making no
                // references to objects inside, later causes an exception in the code.
                // Very strange
                //            if unloadBundle { bundle.unload() }
            }
        } catch {
            writeLog(String(describing: error))
        }
        Platform._services = loadedServices
    }
    
    //--------------------------------------------------------------------------
    /// getPlugInBundles
    /// an array of the dynamically installed bundles
    private func getPlugInBundles() -> [Bundle] {
        if let dir = Bundle.main.builtInPlugInsPath {
            return Bundle.paths(forResourcesOfType: "bundle", inDirectory: dir)
                .map { Bundle(url: URL(fileURLWithPath: $0))! }
        } else {
            return []
        }
    }
    
    //--------------------------------------------------------------------------
    // defaultDevice
    // selects a ComputeDevice based on `servicePriority` and
    // `deviceIdPriority`. It is guaranteed that at least one device like
    // the cpu is available
    var defaultDevice: ComputeDevice {
        guard _defaultDevice == nil else { return _defaultDevice! }
        
        // try to exact match the service request
        let requestedDevice = deviceIdPriority[0]
        for serviceName in servicePriority where _defaultDevice == nil {
            _defaultDevice = requestDevice(serviceName: serviceName,
                                           deviceId: requestedDevice)
        }
        
        // if the search failed, then use the cpu
        _defaultDevice = _defaultDevice ?? requestDevice()
        let device = _defaultDevice!
        writeLog("default device: [\(device.service.name)] \(device.name)",
            level: .status)
        return device
    }
    
    //--------------------------------------------------------------------------
    /// requestDevice
    /// - Parameter serviceName: optional (cpu, cuda, tpu, ...)
    /// - Parameter deviceId: selected device id (0, 1, 2, ...)
    /// - Returns: the requested device from the requested service. If the
    /// service or device is not available, then a substrituion will be made
    /// based on Platform `servicePriority`and `deviceIdPriority`. The CPU
    /// is always available.
    func requestDevice(serviceName: String? = nil,
                       deviceId: Int = 0) -> ComputeDevice
    {
        if let name = serviceName {
            if let service = services[name] {
                return service.devices[deviceId % service.devices.count]
            } else {
                writeLog("Service `\(name)` not found. " +
                    "Substituting `\(Platform.cpu.service.name)`",
                    level: .warning)
                return Platform.cpu
            }
        } else {
            return Platform.cpu
        }
    }
    
    //--------------------------------------------------------------------------
    /// open
    /// this is a placeholder. Additional parameters will be needed for
    /// credentials, timeouts, etc...
    ///
    /// - Parameter url: the location of the remote platform
    /// - Returns: a reference to the remote platform, which can be used
    ///   to query resources and create remote queues.
    static func open(platform url: URL) throws -> ComputePlatform {
        fatalError("not implemented yet")
    }
}

