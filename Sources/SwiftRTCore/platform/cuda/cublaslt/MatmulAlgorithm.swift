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
import CCuda

//==============================================================================
// MatmulAlgorithm
public final class MatmulAlgorithm 
{
    public var desc: cublasLtMatmulAlgo_t

    // initializers
    @inlinable public init(
        cublas: CublasHandle,
        computeType: cublasComputeType_t,
        scaleType: cudaDataType_t,
        aType: cudaDataType_t,
        bType: cudaDataType_t,
        cType: cudaDataType_t,
        dType: cudaDataType_t,
        algoId: Int32
    ) {
        desc = cublasLtMatmulAlgo_t()
        cudaCheck(cublasLtMatmulAlgoInit(
            cublas.handle, computeType, scaleType, aType, bType,
            cType, dType, algoId, &desc))
    }

    //--------------------------------------------------------------------------
    /// getAttribute
    @inlinable public func getCap<T>(
        _ attr: cublasLtMatmulAlgoCapAttributes_t, 
        _ value: inout T
    ) {
        var written = 0
        cudaCheck(cublasLtMatmulAlgoCapGetAttribute(
            &desc, attr, &value, MemoryLayout.size(ofValue: value), &written))
    }

    //--------------------------------------------------------------------------
    /// getConfig
    @inlinable public func getConfig<T>(
        _ attr: cublasLtMatmulAlgoConfigAttributes_t, 
        _ value: inout T
    ) {
        var written = 0
        cudaCheck(cublasLtMatmulAlgoConfigGetAttribute(
            &desc, attr, &value, MemoryLayout.size(ofValue: value), &written))
    }

    /// setConfig
    @inlinable public func setConfig<T>(
        _ attr: cublasLtMatmulAlgoConfigAttributes_t,
         _ value: inout T
    ) {
        cudaCheck(cublasLtMatmulAlgoConfigSetAttribute(
            &desc, attr, &value, MemoryLayout.size(ofValue: value)))
    }

    //==========================================================================
    // Config properties
    //==========================================================================

    //--------------------------------------------------------------------------
    /// algorithm index
    @inlinable public var id: Int {
        var value: Int32 = 0
        getConfig(CUBLASLT_ALGO_CONFIG_ID, &value)
        return Int(value)
    }

    //--------------------------------------------------------------------------
    /// Support for split-K. See 
    @inlinable public var tileId: MatmulTile {
        get {
            var value = CUBLASLT_MATMUL_TILE_UNDEFINED
            getConfig(CUBLASLT_ALGO_CONFIG_TILE_ID, &value)
            return MatmulTile(value)
        }
        set {
            var value = newValue.cublas
            setConfig(CUBLASLT_ALGO_CONFIG_TILE_ID, &value)
        }
    }

    //==========================================================================
    // Caps properties
    //==========================================================================

    //--------------------------------------------------------------------------
    /// Support for split-K. See 
    @inlinable public var supportsSplitK: Bool {
        var value: Int32 = 0
        getCap(CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &value)
        return value == 1
    }

}

//==============================================================================
/// MatmulTile
/// Tile size (in C/D matrix Rows x Cols)
/// General order of tile IDs is sorted by size first and by 
/// first dimension second.
public enum MatmulTile {
    case undefined
    case tile8x8
    case tile8x16
    case tile16x8
    case tile8x32
    case tile16x16
    case tile32x8
    case tile8x64
    case tile16x32
    case tile32x16
    case tile64x8
    case tile32x32
    case tile32x64
    case tile64x32
    case tile32x128
    case tile64x64
    case tile128x32
    case tile64x128
    case tile128x64
    case tile64x256
    case tile128x128
    case tile256x64
    case tile64x512
    case tile128x256
    case tile256x128
    case tile512x64
}

extension MatmulTile {
    @inlinable public init(_ type: cublasLtMatmulTile_t) {
        switch type {
        case CUBLASLT_MATMUL_TILE_UNDEFINED: self = .undefined
        case CUBLASLT_MATMUL_TILE_8x8: self = .tile8x8
        case CUBLASLT_MATMUL_TILE_8x16: self = .tile8x16
        case CUBLASLT_MATMUL_TILE_16x8: self = .tile16x8
        case CUBLASLT_MATMUL_TILE_8x32: self = .tile8x32
        case CUBLASLT_MATMUL_TILE_16x16: self = .tile16x16
        case CUBLASLT_MATMUL_TILE_32x8: self = .tile32x8
        case CUBLASLT_MATMUL_TILE_8x64: self = .tile8x64
        case CUBLASLT_MATMUL_TILE_16x32: self = .tile16x32
        case CUBLASLT_MATMUL_TILE_32x16: self = .tile32x16
        case CUBLASLT_MATMUL_TILE_64x8: self = .tile64x8
        case CUBLASLT_MATMUL_TILE_32x32: self = .tile32x32
        case CUBLASLT_MATMUL_TILE_32x64: self = .tile32x64
        case CUBLASLT_MATMUL_TILE_64x32: self = .tile64x32
        case CUBLASLT_MATMUL_TILE_32x128: self = .tile32x128
        case CUBLASLT_MATMUL_TILE_64x64: self = .tile64x64
        case CUBLASLT_MATMUL_TILE_128x32: self = .tile128x32
        case CUBLASLT_MATMUL_TILE_64x128: self = .tile64x128
        case CUBLASLT_MATMUL_TILE_128x64: self = .tile128x64
        case CUBLASLT_MATMUL_TILE_64x256: self = .tile64x256
        case CUBLASLT_MATMUL_TILE_128x128: self = .tile128x128
        case CUBLASLT_MATMUL_TILE_256x64: self = .tile256x64
        case CUBLASLT_MATMUL_TILE_64x512: self = .tile64x512
        case CUBLASLT_MATMUL_TILE_128x256: self = .tile128x256
        case CUBLASLT_MATMUL_TILE_256x128: self = .tile256x128
        case CUBLASLT_MATMUL_TILE_512x64: self = .tile512x64
        default: fatalError("unrecognized cublasLtMatmulTile_t")
        }
    }

    @inlinable public var cublas: cublasLtMatmulTile_t {
        let types: [MatmulTile: cublasLtMatmulTile_t] = [
            .undefined: CUBLASLT_MATMUL_TILE_UNDEFINED,
            .tile8x8: CUBLASLT_MATMUL_TILE_8x8,
            .tile8x16: CUBLASLT_MATMUL_TILE_8x16,
            .tile16x8: CUBLASLT_MATMUL_TILE_16x8,
            .tile8x32: CUBLASLT_MATMUL_TILE_8x32,
            .tile16x16: CUBLASLT_MATMUL_TILE_16x16,
            .tile32x8: CUBLASLT_MATMUL_TILE_32x8,
            .tile8x64: CUBLASLT_MATMUL_TILE_8x64,
            .tile16x32: CUBLASLT_MATMUL_TILE_16x32,
            .tile32x16: CUBLASLT_MATMUL_TILE_32x16,
            .tile64x8: CUBLASLT_MATMUL_TILE_64x8,
            .tile32x32: CUBLASLT_MATMUL_TILE_32x32,
            .tile32x64: CUBLASLT_MATMUL_TILE_32x64,
            .tile64x32: CUBLASLT_MATMUL_TILE_64x32,
            .tile32x128: CUBLASLT_MATMUL_TILE_32x128,
            .tile64x64: CUBLASLT_MATMUL_TILE_64x64,
            .tile128x32: CUBLASLT_MATMUL_TILE_128x32,
            .tile64x128: CUBLASLT_MATMUL_TILE_64x128,
            .tile128x64: CUBLASLT_MATMUL_TILE_128x64,
            .tile64x256: CUBLASLT_MATMUL_TILE_64x256,
            .tile128x128: CUBLASLT_MATMUL_TILE_128x128,
            .tile256x64: CUBLASLT_MATMUL_TILE_256x64,
            .tile64x512: CUBLASLT_MATMUL_TILE_64x512,
            .tile128x256: CUBLASLT_MATMUL_TILE_128x256,
            .tile256x128: CUBLASLT_MATMUL_TILE_256x128,
            .tile512x64: CUBLASLT_MATMUL_TILE_512x64
        ]        
        return types[self]!
    }
}