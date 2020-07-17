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
    /// id for tile shape. Default is undefined.
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

    //--------------------------------------------------------------------------
    /// id for tile stages. Default is undefined.
    /// Size and number of stages in which elements are read into shared memory
    @inlinable public var stagesId: MatmulStages {
        get {
            var value = CUBLASLT_MATMUL_STAGES_UNDEFINED
            getConfig(CUBLASLT_ALGO_CONFIG_STAGES_ID, &value)
            return MatmulStages(value)
        }
        set {
            var value = newValue.cublas
            setConfig(CUBLASLT_ALGO_CONFIG_STAGES_ID, &value)
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

//==============================================================================
/// MatmulStages
/// Size and number of stages in which elements are read into shared memory
/// General order of stages IDs is sorted by stage size first and by number
/// of stages second.
public enum MatmulStages {
    case undefined
    case stages16x1
    case stages16x2
    case stages16x3
    case stages16x4
    case stages16x5
    case stages16x6
    case stages32x1
    case stages32x2
    case stages32x3
    case stages32x4
    case stages32x5
    case stages32x6
    case stages64x1
    case stages64x2
    case stages64x3
    case stages64x4
    case stages64x5
    case stages64x6
    case stages128x1
    case stages128x2
    case stages128x3
    case stages128x4
    case stages128x5
    case stages128x6
    case stages32x10
    case stages8x4
    case stages16x10
}

extension MatmulStages {
    @inlinable public init(_ type: cublasLtMatmulStages_t) {
        switch type {
        case CUBLASLT_MATMUL_STAGES_UNDEFINED: self = .undefined
        case CUBLASLT_MATMUL_STAGES_16x1: self = .stages16x1
        case CUBLASLT_MATMUL_STAGES_16x2: self = .stages16x2
        case CUBLASLT_MATMUL_STAGES_16x3: self = .stages16x3
        case CUBLASLT_MATMUL_STAGES_16x4: self = .stages16x4
        case CUBLASLT_MATMUL_STAGES_16x5: self = .stages16x5
        case CUBLASLT_MATMUL_STAGES_16x6: self = .stages16x6
        case CUBLASLT_MATMUL_STAGES_32x1: self = .stages32x1
        case CUBLASLT_MATMUL_STAGES_32x2: self = .stages32x2
        case CUBLASLT_MATMUL_STAGES_32x3: self = .stages32x3
        case CUBLASLT_MATMUL_STAGES_32x4: self = .stages32x4
        case CUBLASLT_MATMUL_STAGES_32x5: self = .stages32x5
        case CUBLASLT_MATMUL_STAGES_32x6: self = .stages32x6
        case CUBLASLT_MATMUL_STAGES_64x1: self = .stages64x1
        case CUBLASLT_MATMUL_STAGES_64x2: self = .stages64x2
        case CUBLASLT_MATMUL_STAGES_64x3: self = .stages64x3
        case CUBLASLT_MATMUL_STAGES_64x4: self = .stages64x4
        case CUBLASLT_MATMUL_STAGES_64x5: self = .stages64x5
        case CUBLASLT_MATMUL_STAGES_64x6: self = .stages64x6
        case CUBLASLT_MATMUL_STAGES_128x1: self = .stages128x1
        case CUBLASLT_MATMUL_STAGES_128x2: self = .stages128x2
        case CUBLASLT_MATMUL_STAGES_128x3: self = .stages128x3
        case CUBLASLT_MATMUL_STAGES_128x4: self = .stages128x4
        case CUBLASLT_MATMUL_STAGES_128x5: self = .stages128x5
        case CUBLASLT_MATMUL_STAGES_128x6: self = .stages128x6
        case CUBLASLT_MATMUL_STAGES_32x10: self = .stages32x10
        case CUBLASLT_MATMUL_STAGES_8x4:   self = .stages8x4
        case CUBLASLT_MATMUL_STAGES_16x10: self = .stages16x10
        default: fatalError("unrecognized cublasLtMatmulStages_t")
        }
    }

    @inlinable public var cublas: cublasLtMatmulStages_t {
        let types: [MatmulStages: cublasLtMatmulStages_t] = [
            .undefined:  CUBLASLT_MATMUL_STAGES_UNDEFINED,
            .stages16x1: CUBLASLT_MATMUL_STAGES_16x1,
            .stages16x2: CUBLASLT_MATMUL_STAGES_16x2,
            .stages16x3: CUBLASLT_MATMUL_STAGES_16x3,
            .stages16x4: CUBLASLT_MATMUL_STAGES_16x4,
            .stages16x5: CUBLASLT_MATMUL_STAGES_16x5,
            .stages16x6: CUBLASLT_MATMUL_STAGES_16x6,
            .stages32x1: CUBLASLT_MATMUL_STAGES_32x1,
            .stages32x2: CUBLASLT_MATMUL_STAGES_32x2,
            .stages32x3: CUBLASLT_MATMUL_STAGES_32x3,
            .stages32x4: CUBLASLT_MATMUL_STAGES_32x4,
            .stages32x5: CUBLASLT_MATMUL_STAGES_32x5,
            .stages32x6: CUBLASLT_MATMUL_STAGES_32x6,
            .stages64x1: CUBLASLT_MATMUL_STAGES_64x1,
            .stages64x2: CUBLASLT_MATMUL_STAGES_64x2,
            .stages64x3: CUBLASLT_MATMUL_STAGES_64x3,
            .stages64x4: CUBLASLT_MATMUL_STAGES_64x4,
            .stages64x5: CUBLASLT_MATMUL_STAGES_64x5,
            .stages64x6: CUBLASLT_MATMUL_STAGES_64x6,
            .stages128x1: CUBLASLT_MATMUL_STAGES_128x1,
            .stages128x2: CUBLASLT_MATMUL_STAGES_128x2,
            .stages128x3: CUBLASLT_MATMUL_STAGES_128x3,
            .stages128x4: CUBLASLT_MATMUL_STAGES_128x4,
            .stages128x5: CUBLASLT_MATMUL_STAGES_128x5,
            .stages128x6: CUBLASLT_MATMUL_STAGES_128x6,
            .stages32x10: CUBLASLT_MATMUL_STAGES_32x10,
            .stages8x4:   CUBLASLT_MATMUL_STAGES_8x4,
            .stages16x10: CUBLASLT_MATMUL_STAGES_16x10,
        ]        
        return types[self]!
    }
}
