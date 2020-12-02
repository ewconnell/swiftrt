
add_library(SwiftRTCoreTests
  XCTestManifests.swift
  test_AlgebraicField.swift
  test_Async.swift
  test_Casting.swift
  test_Codable.swift
  test_Comparative.swift
  test_Fractals.swift
  test_Initialize.swift
  test_Math.swift
  test_PackedElements.swift
  test_Pool.swift
  test_Random.swift
  test_Reductions.swift
  test_Shape.swift
  test_StorageElement.swift
  test_Subscripting.swift
  test_VectorElement.swift
  test_Vectorizing.swift
  test_arraySyntax.swift)
target_link_libraries(SwiftRTCoreTests PUBLIC
  $<$<AND:$<BOOL:Foundation_FOUND>,$<NOT:$<PLATFORM_ID:Darwin>>>:Foundation>
  $<$<BOOL:XCTest_Found>:XCTest>
  SwiftRT)
