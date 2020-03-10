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

//==============================================================================
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
// Cuda service module
if buildCuda {
//    let currentDir = FileManager().currentDirectoryPath
//    let kernelsDir = "\(currentDir)/Sources/SwiftRT/platform/cuda/kernels"
//    let kernelsLibName = "SwiftRTCudaKernels"

    //---------------------------------------
    // add Cuda system module
    products.append(.library(name: "CCuda", targets: ["CCuda"]))
    dependencies.append("CCuda")
    targets.append(
        .systemLibrary(name: "CCuda", path: "Modules/Cuda", pkgConfig: "cuda"))
    
    //---------------------------------------
    // add SwiftRT Cuda kernels library built first via cmake

} else {
    exclusions.append("platform/cuda")
}

//==============================================================================
// package specification
targets.append(
    .target(name: "SwiftRT", dependencies: dependencies, exclude: exclusions))

targets.append(
    .testTarget(name: "SwiftRTTests", dependencies: ["SwiftRT"]))

let package = Package(
    name: "SwiftRT",
    products: products,
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics",
                 .branch("master"))
    ],
    targets: targets
)
