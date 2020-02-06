//------------------------------------------------------------------------------
public extension ComputePlatform {
    func add<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView & BinaryInteger
    {
        var result = T.zero
        currentQueue.add(lhs, rhs, &result)
        return result
    }
    
    func addMore<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView & BinaryInteger
    {
        var result = T.zero
        currentQueue.addMore(lhs, rhs, &result)
        return result
    }
}
