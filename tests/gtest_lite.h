#pragma once

#if defined(USE_SYSTEM_GTEST)
    #include <gtest/gtest.h>
#else
// Self-contained GoogleTest-compatible C++20 verification harness for deterministic offline builds
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gtest_lite {

template <typename T>
inline auto format_val(const T& v) {
    if constexpr (std::is_enum_v<T>) {
        return static_cast<std::underlying_type_t<T>>(v);
    } else {
        return v;
    }
}

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline int& fail_counter() {
    static int c = 0;
    return c;
}

struct Registrar {
    Registrar(const char* s, const char* n, std::function<void()> f) {
        registry().push_back({s, n, std::move(f)});
    }
};

struct FatalAssertionError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void record_failure(const char* file, int line, const std::string& msg, bool fatal) {
    ++fail_counter();
    std::cerr << file << ":" << line << ": Failure\n  " << msg << "\n";
    if (fatal) {
        throw FatalAssertionError(msg);
    }
}

inline int run_all_tests() {
    std::cout << "[==========] Running " << registry().size() << " tests.\n";
    int passed = 0;
    int failed = 0;
    for (const auto& tc : registry()) {
        std::string full = tc.suite + "." + tc.name;
        std::cout << "[ RUN      ] " << full << "\n";
        int before = fail_counter();
        try {
            tc.fn();
        } catch (const FatalAssertionError&) {
        } catch (const std::exception& ex) {
            ++fail_counter();
            std::cerr << "  Uncaught std::exception: " << ex.what() << "\n";
        }
        if (fail_counter() == before) {
            ++passed;
            std::cout << "[       OK ] " << full << "\n";
        } else {
            ++failed;
            std::cout << "[  FAILED  ] " << full << "\n";
        }
    }
    std::cout << "[==========] " << passed << " passed, " << failed << " failed.\n";
    return (failed == 0) ? 0 : 1;
}

} // namespace gtest_lite

#define TEST(Suite, Name)                                                      \
    static void Test_##Suite##_##Name();                                       \
    static ::gtest_lite::Registrar Reg_##Suite##_##Name(#Suite, #Name, Test_##Suite##_##Name); \
    static void Test_##Suite##_##Name()

#define EXPECT_TRUE(cond)                                                      \
    do { if (!(cond)) ::gtest_lite::record_failure(__FILE__, __LINE__, "Expected true: " #cond, false); } while (0)

#define ASSERT_TRUE(cond)                                                      \
    do { if (!(cond)) ::gtest_lite::record_failure(__FILE__, __LINE__, "Assert true: " #cond, true); } while (0)

#define EXPECT_FALSE(cond) EXPECT_TRUE(!(cond))

#define EXPECT_EQ(a, b)                                                        \
    do { auto _va = (a); auto _vb = (b);                                       \
         if (!(_va == _vb)) {                                                  \
             std::ostringstream _os; _os << #a " == " #b " (" << ::gtest_lite::format_val(_va) << " vs " << ::gtest_lite::format_val(_vb) << ")"; \
             ::gtest_lite::record_failure(__FILE__, __LINE__, _os.str(), false); \
         } } while (0)

#define ASSERT_EQ(a, b)                                                        \
    do { auto _va = (a); auto _vb = (b);                                       \
         if (!(_va == _vb)) {                                                  \
             std::ostringstream _os; _os << #a " == " #b " (" << ::gtest_lite::format_val(_va) << " vs " << ::gtest_lite::format_val(_vb) << ")"; \
             ::gtest_lite::record_failure(__FILE__, __LINE__, _os.str(), true); \
         } } while (0)

#define EXPECT_GE(a, b)                                                        \
    do { auto _va = (a); auto _vb = (b);                                       \
         if (!(_va >= _vb)) {                                                  \
             std::ostringstream _os; _os << #a " >= " #b " (" << _va << " vs " << _vb << ")"; \
             ::gtest_lite::record_failure(__FILE__, __LINE__, _os.str(), false); \
         } } while (0)

#define EXPECT_LE(a, b)                                                        \
    do { auto _va = (a); auto _vb = (b);                                       \
         if (!(_va <= _vb)) {                                                  \
             std::ostringstream _os; _os << #a " <= " #b " (" << _va << " vs " << _vb << ")"; \
             ::gtest_lite::record_failure(__FILE__, __LINE__, _os.str(), false); \
         } } while (0)

#define EXPECT_GT(a, b)                                                        \
    do { auto _va = (a); auto _vb = (b);                                       \
         if (!(_va > _vb)) {                                                   \
             std::ostringstream _os; _os << #a " > " #b " (" << _va << " vs " << _vb << ")"; \
             ::gtest_lite::record_failure(__FILE__, __LINE__, _os.str(), false); \
         } } while (0)

#define EXPECT_LT(a, b)                                                        \
    do { auto _va = (a); auto _vb = (b);                                       \
         if (!(_va < _vb)) {                                                   \
             std::ostringstream _os; _os << #a " < " #b " (" << _va << " vs " << _vb << ")"; \
             ::gtest_lite::record_failure(__FILE__, __LINE__, _os.str(), false); \
         } } while (0)

#define EXPECT_NEAR(a, b, tol)                                                 \
    do { double _va = static_cast<double>(a); double _vb = static_cast<double>(b); double _t = static_cast<double>(tol); \
         if (std::fabs(_va - _vb) > _t) {                                      \
             std::ostringstream _os; _os << "|" #a " - " #b "| <= " << _t << " (" << _va << " vs " << _vb << ")"; \
             ::gtest_lite::record_failure(__FILE__, __LINE__, _os.str(), false); \
         } } while (0)

#endif
