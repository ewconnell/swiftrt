// By Howard Hinnant from github
#include <cstring>
#include <string>
#include <string_view>
#include <typeinfo>
#include <type_traits>

#if __has_include(<cxxabi.h>)
#   include <cxxabi.h>
#   include <cstdlib>
#   include <memory>
#endif

template <typename T>
std::string type_name() {
    using TR = typename std::remove_reference<T>::type;
    std::unique_ptr<char, void(*)(void*)> own (
    #ifndef _MSC_VER
        abi::__cxa_demangle (typeid(TR).name(), nullptr,
            nullptr, nullptr),
    #else
            nullptr,
    #endif
            std::free
    );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}
