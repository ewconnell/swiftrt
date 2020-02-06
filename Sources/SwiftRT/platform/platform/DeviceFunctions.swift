//==============================================================================
// DeviceFunctions
public protocol DeviceFunctions {
    ///
    func add<T>(_ lhs: T, _ rhs: T, _ result: inout T)
        where T: TensorView & BinaryInteger
    ///
    func addMore<T>(_ lhs: T, _ rhs: T, _ result: inout T)
        where T: TensorView & BinaryInteger
}

//==============================================================================
// DeviceFunctions default cpu delegates
public extension DeviceFunctions {
    func add<T>(_ lhs: T, _ rhs: T, _ result: inout T)
        where T: TensorView & BinaryInteger
    {
        cpu_add(lhs, rhs, &result)
    }
    
    func addMore<T>(_ lhs: T, _ rhs: T, _ result: inout T)
        where T: TensorView & BinaryInteger
    {
        cpu_addMore(lhs, rhs, &result)
    }
}

//==============================================================================
// DeviceFunctions delegates
public extension DeviceFunctions {
    func cpu_add<T>(_ lhs: T, _ rhs: T, _ result: inout T)
        where T: TensorView & BinaryInteger
    {
        result = lhs + rhs
    }
    
    func cpu_addMore<T>(_ lhs: T, _ rhs: T, _ result: inout T)
        where T: TensorView & BinaryInteger
    {
        let vl = [T](repeating: lhs, count: 10)
        let vr = [T](repeating: rhs, count: 10)
        result = (zip(vl, vr).map { $0 + $1 }).reduce(0,+) / 10
    }
}

