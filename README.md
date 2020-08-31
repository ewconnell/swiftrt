***
# Overview
SwiftRT is an experimental computational framework written in the Swift language. The project is under heavy devopment, so frequent changes should be expected.

***
# Installation
This project requires the use of the Google Swift4 TensorFlow custom toolchain (in place of the standard toolchain), because it leverages the toolchain's integrated auto differentiation functionality.

## MacOS and Ubuntu CPU Only Installation
Currently a cpu build is available on MacOS.

1) Install Xcode 12
2) Install the Google [Swift4 TensorFlow Xcode 12 Toolchain](https://github.com/tensorflow/swift/blob/master/Installation.md), and in Xcode select the toolchain in the `Preferences` pane.

3) Install and run the SwiftRT unit tests to verify the installation:
```sh
$ git clone https://github.com/ewconnell/swiftrt.git
$ cd swiftrt
$ swift test
```
4) To create an Xcode project for debugging
```sh
$ swift package generate-xcodeproj
```

## Ubuntu Cuda Installation
Currently the Swift Package Manager does not support building `.cpp` or cuda `.cu` files, so **CMake** is used. CMake and the SPM are used in combination in order to resolve dependent packages during build.

**NOTE:** all unit tests should pass with the CPU only build. The Cuda version is currently under development and there is no claim that the unit tests currently pass. This notice will be removed after they successfully pass.

1) Install the Google [Swift4 TensorFlow Xcode 12 Toolchain](https://github.com/tensorflow/swift/blob/master/Installation.md).

2) Add the following exports to your `~/.bashrc` file, assuming SwiftRT installation is in the `$HOME` directory. 
```bash
# swiftrt installation location
export SWIFTRT_HOME=$HOME/swiftrt

# add to pkgconfig path location of .pc files so SPM can link to Cuda
export PKG_CONFIG_PATH=$SWIFTRT_HOME/pkgconfig

# add `gyb` utility to path if you will be modifying the SwiftRT codebase
# and changing any .gyb files.
export PATH=$SWIFTRT_HOME/gyb:$PATH

# tells Package.swift to include a dependency and link to cuda
export SWIFTRT_PLATFORM=cuda

# cuda installation location
export CUDA_ROOT=/usr/local/cuda
export LIBRARY_PATH=$CUDA_ROOT/lib64

# to enable LLDB to find Cuda symbols
export C_INCLUDE_PATH=$CUDA_ROOT/include
export CPLUS_INCLUDE_PATH=$CUDA_ROOT/include

```
3) CMake 3.5 or higher is required to build. If your current version is less than 3.5, [download and install](https://cmake.org/download/) the latest version.

4) To configure cmake for a Debug version
```sh
$ cd $SWIFTRT_HOME
$ mkdir build
$ cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-9 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-9 -H$SWIFTRT_HOME -B$SWIFTRT_HOME/build -G Ninja
```
5) To build
```sh
$ cd $SWIFTRT_HOME
$ cmake --build ./build --target SwiftRTTests
```
6) To clean
```sh
$ cd $SWIFTRT_HOME
$ cmake --build ./build --target clean
$ rm -rf .build
```
The Swift Package Manager puts all output files into the hidden directory `$SWIFTRT_HOME/.build`

***
## Setting up VScode
VScode is the only IDE so far that seems to be able to build and debug using the custom S4TF toolchain. 

1) [Install VScode](https://code.visualstudio.com/download)
2) The following extensions seem to work.
* Swift Language (Martin Kase)
* CMake Tools (Microsoft)
* CodeLLDB (Vadim Chugunov)
    - It is very important that `settings.json` contains the following entry to pickup the correct lldb version from the toolchain. Substituting `PathToSwiftToolchain` with wherever you installed the toolchain.
    ```sh
    {
        "lldb.library": "PathToSwiftToolchain/usr/lib/liblldb.so"
    }
    ```
* SourceKit-LSP (Pavel Vasek)
    - There is a version of the server as part of the toolchain already, so you don't need to build it. Make sure to configure the extension
    ```sh
    "sourcekit-lsp.serverPath": "PathToSwiftToolchain/usr/bin/sourcekit-lsp"
    ```
* Subtle Match Brackets (Rafa Mel)
* vscode-cudacpp (kriegalex)

2) The `$SWIFTRT_HOME/Documents/vscode` directory contains some helpful starter examples (launch and tasks) to configure the development environment.