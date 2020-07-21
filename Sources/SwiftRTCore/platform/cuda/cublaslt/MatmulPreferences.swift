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
/// MatmulPreferences
public final class MatmulPreferences: CustomStringConvertible
{
    public var desc: cublasLtMatmulPreference_t

    //--------------------------------------------------------------------------
    // initializers
    @inlinable public init(_ desc: cublasLtMatmulPreference_t) {
        self.desc = desc
    }

    /// init
    /// Creates a MatmulPreferences object with default values
    @inlinable public init() {
        var temp: cublasLtMatmulPreference_t?
        cudaCheck(cublasLtMatmulPreferenceCreate(&temp))
        self.desc = temp!

        cudaCheck(cublasLtMatmulPreferenceInit(desc))
    }

    @inlinable deinit {
        cudaCheck(cublasLtMatmulPreferenceDestroy(desc))
    }

    //--------------------------------------------------------------------------
    @inlinable public var description: String {
        ""
    }

    //--------------------------------------------------------------------------
    /// getAttribute
    @inlinable public func getAttribute<T>(
        _ attr: cublasLtMatmulPreferenceAttributes_t, 
        _ value: inout T
    ) {
        var written = 0
        cudaCheck(cublasLtMatmulPreferenceGetAttribute(
            desc, attr, &value, MemoryLayout.size(ofValue: value), &written))
    }

    /// setAttribute
    @inlinable public func setAttribute<T>(
        _ attr: cublasLtMatmulPreferenceAttributes_t,
         _ value: inout T
    ) {
        cudaCheck(cublasLtMatmulPreferenceSetAttribute(
            desc, attr, &value, MemoryLayout.size(ofValue: value)))
    }

    //--------------------------------------------------------------------------
    /// Search mode. Default is `.bestFit`
    @inlinable public var searchMode: MatmulSearch {
        get {
            var value = cublasLtMatmulSearch_t(0)
            getAttribute(CUBLASLT_MATMUL_PREF_SEARCH_MODE, &value)
            return MatmulSearch(value)
        }
        set {
            var value = newValue.cublas
            setAttribute(CUBLASLT_MATMUL_PREF_SEARCH_MODE, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Maximum allowed workspace memory size. 
    /// Default is 0 (no workspace memory allowed).
    @inlinable public var maxWorkspaceSize: Int {
        get {
            var value: UInt64 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &value)
            return Int(value)
        }
        set {
            var value = UInt64(newValue)
            setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// The reduction schemes allowed. Default is `.all`
    @inlinable public var reductionSchemes: MatmulReductionSchemeOptions {
        get {
            var value: UInt32 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &value)
            return MatmulReductionSchemeOptions(rawValue: value)
        }
        set {
            var value = newValue.rawValue
            setAttribute(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Minimum buffer alignment for tensor (in bytes). Selecting a
    /// smaller value will exclude algorithms that can not work with the
    /// tensor, which is not as strictly aligned as the algorithms need. 
    /// Default is 256 bytes.
    @inlinable public var minAlignmentA: Int {
        get {
            var value: UInt32 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &value)
            return Int(value)
        }
        set {
            var value = UInt32(newValue)
            setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Minimum buffer alignment for tensor (in bytes). Selecting a
    /// smaller value will exclude algorithms that can not work with the
    /// tensor, which is not as strictly aligned as the algorithms need. 
    /// Default is 256 bytes.
    @inlinable public var minAlignmentB: Int {
        get {
            var value: UInt32 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &value)
            return Int(value)
        }
        set {
            var value = UInt32(newValue)
            setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Minimum buffer alignment for tensor (in bytes). Selecting a
    /// smaller value will exclude algorithms that can not work with the
    /// tensor, which is not as strictly aligned as the algorithms need. 
    /// Default is 256 bytes.
    @inlinable public var minAlignmentC: Int {
        get {
            var value: UInt32 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &value)
            return Int(value)
        }
        set {
            var value = UInt32(newValue)
            setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Minimum buffer alignment for tensor (in bytes). Selecting a
    /// smaller value will exclude algorithms that can not work with the
    /// tensor, which is not as strictly aligned as the algorithms need. 
    /// Default is 256 bytes.
    @inlinable public var minAlignmentD: Int {
        get {
            var value: UInt32 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &value)
            return Int(value)
        }
        set {
            var value = UInt32(newValue)
            setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Maximum wave count. See `MatmulHeuristicResult.waves`. 
    /// Selecting a non-zero value will exclude algorithms that report
    /// device utilization higher than specified. Default is 0, which
    /// will not exclude any alogirthms based on this attribute.
    @inlinable public var maxWaves: Float {
        get {
            var value: Float = 0
            getAttribute(CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT, &value)
            return value
        }
        set {
            var value = newValue
            setAttribute(CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// The reduction schemes allowed. Default is `.all`
    @inlinable public var pointerModeOptions: MatmulReductionSchemeOptions {
        get {
            var value: UInt32 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &value)
            return MatmulReductionSchemeOptions(rawValue: value)
        }
        set {
            var value = newValue.rawValue
            setAttribute(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Epilogue selector mask. Filters the heuristic result to include
    /// only algorithms that support all required operations. 
    @inlinable public var epilogueOptions: MatmulEpilogueOptions {
        get {
            var value: UInt32 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &value)
            return MatmulEpilogueOptions(rawValue: value)
        }
        set {
            var value = newValue.rawValue
            setAttribute(CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Numerical implementation options. See `MatmulNumericalImplementation`
    /// Filters heuristic result to only include algorithms that use the 
    /// allowed implementations. Default is `.all`
    @inlinable public var numericalOptions: MatmulNumericalOptions {
        get {
            var value: UInt64 = 0
            getAttribute(CUBLASLT_MATMUL_PREF_IMPL_MASK, &value)
            return MatmulNumericalOptions(rawValue: value)
        }
        set {
            var value = newValue.rawValue
            setAttribute(CUBLASLT_MATMUL_PREF_IMPL_MASK, &value)
        }
    }
}

//==============================================================================
/// MatmulSearch
public enum MatmulSearch {
    /// Request heuristics for the best algorithm for the given use case.
    case bestFit
    /// Request heuristics only for the pre-configured algo id.
    case limitedByAlgoId
}

extension MatmulSearch {
    @inlinable public init(_ type: cublasLtMatmulSearch_t) {
        switch type {
        case CUBLASLT_SEARCH_BEST_FIT: self = .bestFit
        case CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID: self = .limitedByAlgoId
        default: fatalError("unrecognized cublasLtMatmulSearch_t")
        }
    }

    @inlinable public var cublas: cublasLtMatmulSearch_t {
        let types: [MatmulSearch: cublasLtMatmulSearch_t] = [
            .bestFit: CUBLASLT_SEARCH_BEST_FIT,
            .limitedByAlgoId: CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID,
        ]        
        return types[self]!
    }
}
