#include <stdint.h>

#if (__cplusplus >= 202302L) && __has_include(<stdfloat>)
#include <stdfloat>
#endif

#ifndef __STDCPP_FLOAT64_T__
using float64_t = double;
#endif

#ifndef __STDCPP_FLOAT32_T__
using float32_t = float;
#endif

#ifdef __STDCPP_FLOAT16_T__

using float16m10e5s1_t = float16_t;

#else // No __STDCPP_FLOAT16_T__

// Minimal implementation of float16, for both IEEE's m10e5s1 and Google Brain's m7e8s1.
// See http://half.sourceforge.net/ if you want something mature, as these helpers
// don't support math (addition, multiplication...), just conversion to and from float32.

// Standard IEEE 16-bit floating type.
// This class contains just enough logic for conversion, not general math.
struct float16m10e5s1_t
{
    float16m10e5s1_t() = default;
    float16m10e5s1_t(const float16m10e5s1_t&) = default;
    float16m10e5s1_t(float16m10e5s1_t&&) = default;

    static uint32_t constexpr float16mantissaCount = 10;
    static uint32_t constexpr float32mantissaCount = 23;
    static int32_t  constexpr float32to16mantissaCountDifference = float32mantissaCount - float16mantissaCount;
    static uint32_t constexpr float16exponentCount = 5;
    static uint32_t constexpr float32exponentCount = 8;
    static int32_t  constexpr float16exponentBias = 15;
    static int32_t  constexpr float16exponentMin = 0; // effectively 2^(0-15)
    static int32_t  constexpr float16exponentMax = 31; // effectively 2^(31-15) or 2^16
    static int32_t  constexpr float32exponentBias = 127;
    static int32_t  constexpr float32exponentMin = 0; // effectively 2^(0-127)
    static int32_t  constexpr float32exponentMax = 255; // effectively 2^(255-127) or 2^128
    static int32_t  constexpr float32vs16exponentAdjustment = float32exponentBias - float16exponentBias;
    static uint32_t constexpr float16signMask = 0b1'00000'0000000000;
    static uint32_t constexpr float32signMask = 0b10000000'00000000'00000000'00000000;
    static uint32_t constexpr float16mantissaMask = 0b0'00000'1111111111;
    static uint32_t constexpr float16exponentMask = 0b0'11111'0000000000;
    static uint32_t constexpr float16mantissaAndExponentMask = float16mantissaMask | float16exponentMask;
    static uint32_t constexpr float32mantissaMask = 0b00000000'01111111'11111111'11111111;
    static uint32_t constexpr float32exponentMask = 0b01111111'10000000'00000000'00000000;
    static uint32_t constexpr float32mantissaAndExponentMask = float32mantissaMask | float32exponentMask;
    static uint32_t constexpr float32minimum16bitExponent = (float16exponentMin + float32vs16exponentAdjustment) << float32mantissaCount;
    static uint32_t constexpr float32maximum16bitExponent = (float16exponentMax + float32vs16exponentAdjustment) << float32mantissaCount;

    float16m10e5s1_t(float floatValue) noexcept
    {
        // Shift the mantissa, exponent, and sign from the 32-bit locations to 16-bit.
        // Sature the exponent if greater than float16 can represent.
        // float32 denorms are flushed to zero.
        uint32_t const float32bitValue = reinterpret_cast<uint32_t&>(floatValue);
        uint32_t const sign = (float32bitValue >> 16) & float16signMask;
        int32_t  const float32mantissaAndExponent = float32bitValue & float32mantissaAndExponentMask; // Keep every bit except the sign.
        int32_t  const float16mantissaAndExponent = (float32mantissaAndExponent >> float32to16mantissaCountDifference) - (float32vs16exponentAdjustment << float16mantissaCount); // Adjust the bits and exponent range.
        uint32_t const float16denormMask = (float16mantissaAndExponent > int32_t(float16mantissaMask)) ? float16mantissaAndExponentMask : 0;
        uint32_t const float16saturationMask = (float16mantissaAndExponent >= int32_t(float16mantissaAndExponentMask)) ? float16exponentMask : 0;
        uint32_t const float16bitValue = (float16mantissaAndExponent & float16denormMask) | float16saturationMask | sign;
        value = uint16_t(float16bitValue);
    }

    float16m10e5s1_t& operator =(const float16m10e5s1_t&) = default;

    inline float16m10e5s1_t& operator =(float floatValue) noexcept
    {
        new(this) float16m10e5s1_t(floatValue);
        return *this;
    }

    operator float() const noexcept
    {
        // Shift the mantissa, exponent, and sign from the 16-bit locations to 32-bit.
        // Saturate the exponent to positive infinity for float32 if the float16 was infinity.
        // float32 denorms are flushed to zero.
        uint32_t const float16bitValue = value;
        uint32_t const sign = (float16bitValue << 16) & float32signMask;
        int32_t  const float16mantissaAndExponent = float16bitValue & float16mantissaAndExponentMask; // Keep every bit except the sign.
        int32_t  const float32mantissaAndExponent = (float16mantissaAndExponent << float32to16mantissaCountDifference) + (float32vs16exponentAdjustment << float32mantissaCount);
        uint32_t const float32denormMask = (float32mantissaAndExponent > int32_t(float32minimum16bitExponent | float32mantissaMask)) ? float32mantissaAndExponentMask : 0;
        uint32_t const float32saturationMask = (float32mantissaAndExponent >= int32_t(float32maximum16bitExponent)) ? float32exponentMask : 0;
        uint32_t const float32bitValue = (float32mantissaAndExponent & float32denormMask) | float32saturationMask | sign;
        float floatValue = 0.0;
        reinterpret_cast<uint32_t&>(floatValue) = float32bitValue;
        return floatValue;
    }

    uint16_t value;

    // constexpr float testNumbers[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 65504.0f, -65504.0f, 16777216.0f, -16777216.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity()};
};

#endif // __STDCPP_FLOAT16_T__


// Truncated IEEE 32-bit floating type to 16-bits (so called "brain float").
struct float16m7e8s1_t
{
    float16m7e8s1_t() = default;
    float16m7e8s1_t(const float16m7e8s1_t&) = default;
    float16m7e8s1_t(float16m7e8s1_t&&) = default;

    float16m7e8s1_t(float floatValue) noexcept
    {
        value = reinterpret_cast<uint32_t&>(floatValue) >> 16;
    }

    float16m7e8s1_t& operator =(const float16m7e8s1_t&) = default;

    inline float16m7e8s1_t& operator =(float floatValue) noexcept
    {
        new(this) float16m7e8s1_t(floatValue);
        return *this;
    }

    operator float() const noexcept
    {
        float floatValue = 0.0;
        reinterpret_cast<uint32_t&>(floatValue) = value << 16;
        return floatValue;
    }

    uint16_t value;
};
