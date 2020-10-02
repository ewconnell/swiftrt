add_library(SwiftRTLayers
  Convolution.swift
  Dense.swift
  Embedding.swift
  Layer.swift
  Normalization.swift
  ParameterInitializer.swift
  Recurrent.swift
  common/BijectiveDictionary.swift)
set_target_properties(SwiftRTLayers PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(SwiftRTLayers PUBLIC
  SwiftRTCore)
