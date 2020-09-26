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

//==============================================================================
/// MatmulAlgorithm
public final class MatmulAlgorithm: CustomStringConvertible
{
    public var desc: cublasLtMatmulAlgo_t

    //--------------------------------------------------------------------------
    // initializers
    @inlinable public init(_ desc: cublasLtMatmulAlgo_t) {
        self.desc = desc
    }

    @inlinable public init(
        algoId: Int,
        accumulatorType: MatmulAccumulatorType,
        scaleType: srtDataType,
        aType: srtDataType,
        bType: srtDataType,
        cType: srtDataType,
        dType: srtDataType,
        using queue: Platform.Device.Queue = currentQueue
    ) {
        assert(cType == dType, "must be equal for now")
        desc = cublasLtMatmulAlgo_t()
        cudaCheck(cublasLtMatmulAlgoInit(
            queue.cublas.handle, 
            accumulatorType.cublas, 
            cudaDataType(scaleType), 
            cudaDataType(aType), cudaDataType(bType),
            cudaDataType(cType), cudaDataType(dType), 
            Int32(algoId), &desc))
    }

    //--------------------------------------------------------------------------
    /// getIds
    ///
    /// Parameters:
    ///  - queue: the device queue to use
    ///  - 
    /// Returns: an array of algorithm ids that match the specified requirements
    public static func getIds(
        maxIds: Int,
        accumulatorType: MatmulAccumulatorType,
        scaleType: srtDataType,
        aType: srtDataType,
        bType: srtDataType,
        cType: srtDataType,
        dType: srtDataType,
        using queue: Platform.Device.Queue = currentQueue
    ) -> [Int] {
        assert(cType == dType, "must be equal for now")
        var tempIds = [Int32](repeating: 0, count: maxIds)
        var tempFound: Int32 = 0
        cudaCheck(cublasLtMatmulAlgoGetIds(
            queue.cublas.handle, 
            accumulatorType.cublas,
            cudaDataType(scaleType),
            cudaDataType(aType), cudaDataType(bType),
            cudaDataType(cType), cudaDataType(dType),
            Int32(maxIds),
            &tempIds, &tempFound))
        return tempIds[0..<Int(tempFound)].map(Int.init)
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

    //--------------------------------------------------------------------------
    /// description
    public var description: String {
        """
        MatmulAlgorithm configuration
        id             : \(id)
        tileId         : \(tileId)
        stagesId       : \(stagesId)
        splitK         : \(splitK)
        reductionScheme: \(reductionScheme)
        threadSwizzle  : \(threadSwizzle)
        customOption   : \(customOption)
        """
    }

    /// capsDescription
    public var capsDescription: String {
        """
        MatmulAlgorithm caps
        supportsSplitK            : \(supportsSplitK)
        threadSwizzling           : \(threadSwizzling)
        supportsStridedBatching   : \(supportsStridedBatching)
        supportsOutOfPlaceResult  : \(supportsOutOfPlaceResult)
        supportsUPLO              : \(supportsUPLO)
        tileIds                   : \(tileIds.map { "\(MatmulTile($0))" })
        stageIds                  : \(stagesIds.map { "\(MatmulStages($0))" })
        customOptionCount         : \(customOptionCount)
        customMemoryOrder         : \(customMemoryOrder)
        supportsNegativeLeadingDim: \(supportsNegativeLeadingDim)
        numericalImplementation   : \(numericalImplementation)
        alignmentA                : \(alignmentA)
        alignmentB                : \(alignmentB)
        alignmentC                : \(alignmentC)
        alignmentD                : \(alignmentD)
        """
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

    //--------------------------------------------------------------------------
    /// Number of K splits. If != 1, then K split parts of matrix multiplication
    /// will be computed in parallel, and then the results accumulated 
    /// according to the `reductionScheme`
    @inlinable public var splitK: Int {
        get {
            var value: UInt32 = 1
            getConfig(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &value)
            return Int(value)
        }
        set {
            var value = UInt32(newValue)
            setConfig(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Reduction scheme to use when `splitK` value > 1
    @inlinable public var reductionScheme: MatmulReductionScheme {
        get {
            var value: UInt32 = 0
            assert(MemoryLayout<cublasLtReductionScheme_t>.size ==
                   MemoryLayout.size(ofValue: value))
            getConfig(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &value)
            return MatmulReductionScheme(value)
        }
        set {
            var value = newValue.cublas
            setConfig(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Enable/Disable CTA swizzling. Change mapping from grid
    /// coordinates to parts of the matrices.
    @inlinable public var threadSwizzle: MatmulThreadSwizzling {
        get {
            var value: UInt32 = 0
            getConfig(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &value)
            return MatmulThreadSwizzling(rawValue: value)!
        }
        set {
            var value: UInt32 = newValue.rawValue
            setConfig(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Custom option value. Each algorithm can support some custom options
    /// that don't fit the description of the other configuration attributes.
    /// The valid range is 0...caps.customOptionMax
    @inlinable public var customOption: Int {
        get {
            var value: UInt32 = 0
            getConfig(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &value)
            return Int(value)
        }
        set {
            var value = UInt32(newValue)
            setConfig(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &value)
        }
    }

    //==========================================================================
    // Caps properties
    //==========================================================================

    //--------------------------------------------------------------------------
    /// Supports split-K 
    @inlinable public var supportsSplitK: Bool {
        var value: Int32 = 0
        getCap(CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &value)
        return value == 1
    }

    //--------------------------------------------------------------------------
    /// The number of cooperative thread array swizzling variations
    @inlinable public var threadSwizzling: MatmulThreadSwizzling {
        var value: UInt32 = 0
        getCap(CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &value)
        return MatmulThreadSwizzling(rawValue: value)!
    }

    //--------------------------------------------------------------------------
    /// Supports strided batching
    @inlinable public var supportsStridedBatching: Bool {
        var value: Int32 = 0
        getCap(CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT, &value)
        return value == 1
    }

    //--------------------------------------------------------------------------
    /// Supports results out of place (D != C in D = alpha.A.B + beta.C)
    @inlinable public var supportsOutOfPlaceResult: Bool {
        var value: Int32 = 0
        getCap(CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT, &value)
        return value == 1
    }

    //--------------------------------------------------------------------------
    /// Supports syrk (symmetric rank k update)/herk (Hermitian rank k update)
    /// on top of regular gemm
    @inlinable public var supportsUPLO: Bool {
        var value: UInt32 = 0
        getCap(CUBLASLT_ALGO_CAP_UPLO_SUPPORT, &value)
        return value == 1
    }

    //--------------------------------------------------------------------------
    /// supported tile configurations
    @inlinable public var tileIds: [Int] {
        // query the count
        var bufferSize = 0
        cudaCheck(cublasLtMatmulAlgoCapGetAttribute(
            &desc, CUBLASLT_ALGO_CAP_TILE_IDS, nil, 0, &bufferSize))
        let count = Int(bufferSize) / MemoryLayout<UInt32>.size

        if count == 0 {
            return [Int(unsafeBitCast(CUBLASLT_MATMUL_TILE_UNDEFINED, 
                                      to: UInt32.self))]
        } else {
            var written = 0
            var ids = [UInt32](repeating: 0, count: count)
            cudaCheck(cublasLtMatmulAlgoCapGetAttribute(
                &desc, CUBLASLT_ALGO_CAP_TILE_IDS, &ids, 
                bufferSize, &written))
            return ids.map(Int.init)
        }
    }

    //--------------------------------------------------------------------------
    /// supported stage configuration
    @inlinable public var stagesIds: [Int] {
        // query the count
        var bufferSize = 0
        cudaCheck(cublasLtMatmulAlgoCapGetAttribute(
            &desc, CUBLASLT_ALGO_CAP_STAGES_IDS, nil, 0, &bufferSize))
        let count = Int(bufferSize) / MemoryLayout<UInt32>.size

        if count == 0 {
            return [Int(unsafeBitCast(CUBLASLT_MATMUL_STAGES_UNDEFINED, 
                                      to: UInt32.self))]
        } else {
            var written = 0
            var ids = [UInt32](repeating: 0, count: Int(count))
            cudaCheck(cublasLtMatmulAlgoCapGetAttribute(
                &desc, CUBLASLT_ALGO_CAP_STAGES_IDS, &ids, 
                bufferSize, &written))
            return ids.map(Int.init)
        }
    }

    //--------------------------------------------------------------------------
    /// The number of custom options the algorithm supports
    @inlinable public var customOptionCount: Int {
        var value: Int32 = 0
        getCap(CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &value)
        return Int(value)
    }

    //--------------------------------------------------------------------------
    /// Indicates whether the algorithm supports custom (not COL or 
    /// ROW memory order). `false` means only COL and ROW memory order
    /// is allowed, non-zero means that algo might have different requirements.
    @inlinable public var customMemoryOrder: Bool {
        var value: Int32 = 0
        getCap(CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER, &value)
        return value == 1
    }

    //--------------------------------------------------------------------------
    /// Supports negative leading dimensions
    @inlinable public var supportsNegativeLeadingDim: Bool {
        var value: UInt32 = 0
        getCap(CUBLASLT_ALGO_CAP_LD_NEGATIVE, &value)
        return value == 1
    }

    //--------------------------------------------------------------------------
    /// details about algorithm's implementation that affect it's numerical
    /// behavior
    @inlinable public var numericalImplementation: MatmulNumericalOptions {
        var value: UInt64 = 0
        getCap(CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &value)
        return MatmulNumericalOptions(rawValue: value)
    }

    //--------------------------------------------------------------------------
    /// minimum alignment required for the matrix in bytes
    @inlinable public var alignmentA: Int {
        var value: UInt32 = 0
        getCap(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, &value)
        return Int(value)
    }

    //--------------------------------------------------------------------------
    /// minimum alignment required for the matrix in bytes
    @inlinable public var alignmentB: Int {
        var value: UInt32 = 0
        getCap(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES, &value)
        return Int(value)
    }

    //--------------------------------------------------------------------------
    /// minimum alignment required for the matrix in bytes
    @inlinable public var alignmentC: Int {
        var value: UInt32 = 0
        getCap(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, &value)
        return Int(value)
    }

    //--------------------------------------------------------------------------
    /// minimum alignment required for the matrix in bytes
    @inlinable public var alignmentD: Int {
        var value: UInt32 = 0
        getCap(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES, &value)
        return Int(value)
    }
}

//==============================================================================
/// MatmulNumericalOptions
public struct MatmulNumericalOptions: OptionSet, CustomStringConvertible {
    public let rawValue: UInt64

    @inlinable public init(rawValue: UInt64) { self.rawValue = rawValue }
    @inlinable public init(_ value: UInt64) { self.rawValue = value }

    public static let fma = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA)
    public static let hmma = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
    public static let imma = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA)             
    public static let dmma = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA)             

    public static let tensorOpMask = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK)
    public static let opTypeMask = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK)  

    public static let accumulator16F = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F)
    public static let accumulator32F = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F)
    public static let accumulator64F = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F)
    public static let accumulator32I = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I)

    public static let accumulatorTypeMask = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK)

    public static let input16F  = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F)
    public static let input16BF = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF)
    public static let inputTF32 = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)
    public static let input32F  = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F)
    public static let input64F  = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F)
    public static let input8I   = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I)

    public static let inputTypeMask = MatmulNumericalOptions(CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK)

    public var description: String {
        var string = "["

        if rawValue == 0xFFFFFFFFFFFFFFFF {
            string += ".all"
        } else {
            if contains(.fma)  { string += ".fma, " }
            if contains(.hmma) { string += ".hmma, " }
            if contains(.imma) { string += ".imma, " }
            if contains(.dmma) { string += ".dmma, " }

            if contains(.accumulator16F) { string += ".accumulator16F, " }
            if contains(.accumulator32F) { string += ".accumulator32F, " }
            if contains(.accumulator64F) { string += ".accumulator64F, " }
            if contains(.accumulator32I) { string += ".accumulator32I, " }

            if contains(.input16F)  { string += ".input16F, " }
            if contains(.input16BF) { string += ".input16BF, " }
            if contains(.inputTF32) { string += ".inputTF32, " }
            if contains(.input32F)  { string += ".input32F, " }
            if contains(.input64F)  { string += ".input64F, " }
            if contains(.input8I)   { string += ".input8I, " }

            // trim
            if let index = string.lastIndex(of: ",") {
                string = String(string[..<index])
            }
        }
        return string + "]"            
    }
}

//==============================================================================
/// MatmulThreadSwizzling
public enum MatmulThreadSwizzling: UInt32, CaseIterable {
    case disabled = 0
    case enabled = 1
}

//==============================================================================
/// MatmulReductionScheme
public enum MatmulReductionScheme: CaseIterable {
    /// Do not apply reduction. The dot-product will be performed in one sequence.
    case none
    /// Reduction is performed "in place" using the output buffer, parts
    /// are added up in the output data type. Workspace is only used for
    /// counters that guarantee sequentiality.
    case inPlace
    /// Reduction done out of place in a user-provided workspace. 
    /// The intermediate results are stored in the compute type in the 
    /// workspace and reduced in a separate step.
    case accumulatorType
    /// Reduction done out of place in a user-provided workspace.
    /// The intermediate results are stored in the output type in the 
    /// workspace and reduced in a separate step.
    case outputType
    /// Allows all reduction schemes.
    case mask
}

extension MatmulReductionScheme {
    @inlinable public init(_ value: UInt32) {
        self.init(unsafeBitCast(value, to:cublasLtReductionScheme_t.self))
    }
    
    @inlinable public init(_ type: cublasLtReductionScheme_t) {
        switch type {
        case CUBLASLT_REDUCTION_SCHEME_NONE: self = .none
        case CUBLASLT_REDUCTION_SCHEME_INPLACE: self = .inPlace
        case CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE: self = .accumulatorType
        case CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE: self = .outputType
        case CUBLASLT_REDUCTION_SCHEME_MASK: self = .mask
        default: fatalError("unrecognized cublasLtReductionScheme_t")
        }
    }

    @inlinable public var cublas: cublasLtReductionScheme_t {
        let types: [MatmulReductionScheme: cublasLtReductionScheme_t] = [
            .none: CUBLASLT_REDUCTION_SCHEME_NONE,
            .inPlace: CUBLASLT_REDUCTION_SCHEME_INPLACE,
            .accumulatorType: CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
            .outputType: CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE,
            .mask: CUBLASLT_REDUCTION_SCHEME_MASK,
        ]        
        return types[self]!
    }
}

//==============================================================================
/// MatmulReductionSchemeOptions
public struct MatmulReductionSchemeOptions: OptionSet, CustomStringConvertible {
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    public init(_ value: cublasLtReductionScheme_t) {
        self.rawValue = value.rawValue
    }
    public let rawValue: UInt32

    public static let inPlace = MatmulReductionSchemeOptions(CUBLASLT_REDUCTION_SCHEME_INPLACE)
    public static let accumulatorType = MatmulReductionSchemeOptions(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
    public static let outputType = MatmulReductionSchemeOptions(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE)
    public static let all = MatmulReductionSchemeOptions(CUBLASLT_REDUCTION_SCHEME_MASK)

    public var description: String {
        var string = "["

        if contains(.all) {
            string += ".all" 
        } else {
            if contains(.inPlace) { string += ".inPlace, " }
            if contains(.accumulatorType) { string += ".accumulatorType, " }
            if contains(.outputType) { string += ".outputType, " }

            // trim
            if let index = string.lastIndex(of: ",") {
                string = String(string[..<index])
            }
        }
        return string + "]"            
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
    @inlinable public init(_ type: Int) {
        self.init(unsafeBitCast(Int32(type), to: cublasLtMatmulTile_t.self))
    }

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
    @inlinable public init(_ type: Int) {
        self.init(unsafeBitCast(Int32(type), to: cublasLtMatmulStages_t.self))
    }

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
