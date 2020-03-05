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
// determine platform build type
let validPlatforms = Set(arrayLiteral: "cpu", "cuda")
let environment = ProcessInfo.processInfo.environment
let platform = (environment["SWIFTRT_PLATFORM"] ?? "cpu").lowercased()
if !validPlatforms.contains(platform) {
    fatalError("valid SWIFTRT_PLATFORM types: \(validPlatforms)")
}

let buildCuda = platform == "cuda"

//---------------------------------------
// the base products, dependencies, and targets
var products: [PackageDescription.Product] = [
    .library(name: "SwiftRT", targets: ["SwiftRT"])
]
var dependencies: [Target.Dependency] = ["Numerics"]
var exclusions: [String] = []
var targets: [PackageDescription.Target] = []

//==============================================================================
// include the Cuda service module
let currentDir = FileManager().currentDirectoryPath
let kernelsDir = "\(currentDir)/Sources/SwiftRT/platform/cuda/kernels"
let kernelsLibName = "SwiftRTCudaKernels"

if buildCuda {
    //---------------------------------------
    // build kernels library
    if #available(OSX 10.13, *) {
        runCMake(args: ["--version"], workingDir: kernelsDir)
    }

    //---------------------------------------
    // add Cuda system module
    products.append(.library(name: "CCuda", targets: ["CCuda"]))
    dependencies.append("CCuda")
    targets.append(
        .systemLibrary(name: "CCuda", path: "Modules/Cuda", pkgConfig: "cuda"))
    
} else {
    exclusions.append("platform/cuda")
}

//------------------------------------------------------------------------------
@available(OSX 10.13, *)
func runCMake(args: [String], workingDir: String) {
    let task = Process()
    task.currentDirectoryURL = URL(fileURLWithPath: workingDir, isDirectory: true)
    task.executableURL = URL(fileURLWithPath: "/usr/local/bin/cmake")
    task.arguments = args
    
    do {
        let outputPipe = Pipe()
        task.standardOutput = outputPipe
        try task.run()
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        if task.terminationStatus == 0 {
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
    .target(name: "SwiftRT", dependencies: dependencies, exclude: exclusions))

targets.append(
    .testTarget(name: "SwiftRTTests", dependencies: ["SwiftRT"]))

let package = Package(
    name: "SwiftRT",
    platforms: [.macOS(.v10_13)],
    products: products,
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics",
                 .branch("master"))
    ],
    targets: targets
)
