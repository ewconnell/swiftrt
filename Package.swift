// swift-tools-version:5.1
// The swift-tools-version declares the minimum version
// of Swift required to build this package.
import PackageDescription
import Foundation

#if os(Linux)
import Glibc
#else
import Darwin.C
#endif

//------------------------------------------------------------------------------
// test for enabled components
func isEnabled(_ id: String) -> Bool { getenv(id) != nil }
let enableCuda = isEnabled("SWIFTRT_ENABLE_CUDA")

//---------------------------------------
// the base products, dependencies, and targets
var products: [PackageDescription.Product] = [
    .library(name: "SwiftRT", type: .dynamic, targets: ["SwiftRT"]),
    // .library(name: "SwiftRT", type: .static, targets: ["SwiftRT"])
]
var dependencies: [Target.Dependency] = ["Numerics"]
var exclusions: [String] = []
var targets: [PackageDescription.Target] = []

//==============================================================================
// include the Cuda service module
if enableCuda {
    //---------------------------------------
    // build kernels library
    if #available(macOS 10.13, *) {
        //        runMakefile(target: ".build/debug/SwiftRTCudaKernels",
        //                    workingDir: "Sources/SwiftRT/device/cuda/kernels")
    } else {
        print("OS version error. blerg...")
    }
    
    //---------------------------------------
    // add Cuda system module
    products.append(.library(name: "CCuda", targets: ["CCuda"]))
    dependencies.append("CCuda")
    targets.append(
        .systemLibrary(name: "CCuda",
                       path: "Modules/Cuda",
                       pkgConfig: "cuda"))

    //---------------------------------------
    // add CudaService
    products.append(.library(name: "CudaService",
                             targets: ["CudaService"]))
    
    targets.append(.target(name: "CudaService",
                           dependencies: ["SwiftRT", "CCuda"],
                           exclude: ["Kernels"]))
}

//------------------------------------------------------------------------------
@available(macOS 10.13, *)
func runMakefile(target: String, workingDir: String) {
    let fileManager = FileManager()
    let task = Process()
    task.currentDirectoryURL = URL(fileURLWithPath:
        "\(fileManager.currentDirectoryPath)/\(workingDir)", isDirectory: true)
    task.executableURL = URL(fileURLWithPath: "/usr/bin/make")
    
    task.arguments = ["TARGET=\"\(target)\""]
    do {
        let outputPipe = Pipe()
        task.standardOutput = outputPipe
        try task.run()
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        if task.terminationStatus != 0 {
            let output = String(decoding: outputData, as: UTF8.self)
            print(output)
        }
    } catch {
        print(error)
    }
}

//==============================================================================
// package specification
targets.append(
    .target(name: "SwiftRT",
            dependencies: dependencies,
            exclude: exclusions))

targets.append(
    .testTarget(name: "SwiftRTTests",
                dependencies: ["SwiftRT"]))

let package = Package(
    name: "SwiftRT",
    products: products,
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics",
                 .branch("master"))
    ],
    targets: targets
)
