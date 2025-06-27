#pragma once

#include "GPUBuffer.hpp"
#include "GPUFunction.hpp"
#include "Shaders.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace psyne {

/**
 * @brief Registers all the shaders in the GPUFunctionRegistry.
 *
 * Any and all shaders should be registered here, as well as given CPU fallback implementations. You
 * could potentially call the registerFunction elsewhere, but this is a good place to keep all the
 * shader registrations.
 *
 * @author jmorgan
 */
inline void registerAllShaders() {

    //-------------------------------------------------------------------------------------------------------
    // l2_distance.comp - Compute shader for calculating the L2 distance between two vectors.
    // Input: std::vector<float> a, std::vector<float> b
    // Output: float (the L2 distance)
    //-------------------------------------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "l2_distance",
        GPUFunction(
            {std::vector<float>{}, std::vector<float>{}}, 0, L2_DISTANCE_COMP_ZST, 256, 1, 1,
            [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                const auto& a = params[0];
                const auto& b = params[1];
                if (a.size() != b.size()) {
                    throw std::runtime_error("l2_distance: Mismatched vector sizes");
                }
                float sum = 0.0f;
                for (size_t i = 0; i < a.size(); i++) {
                    sum += (a[i] - b[i]) * (a[i] - b[i]);
                }
                std::vector<float> result(1);
                result[0] = std::sqrt(sum);
                return result;
            },
            sizeof(float)));

    //--------------------------------------------------------------------------
    // l2_distance_batch.comp - Compute distances from one vector to many
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "l2_distance_batch", GPUFunction(
            {std::vector<float>{}, std::vector<float>{}}, 0, L2_DISTANCE_BATCH_COMP_ZST,
            256, 1, 1,
            [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                const auto& a = params[0];
                const auto& batch_flat = params[1];
                size_t dim = a.size();
                if (dim == 0 || batch_flat.size() % dim != 0)
                    throw std::runtime_error("l2_distance_batch: Invalid batch shape");
                size_t batch_size = batch_flat.size() / dim;
                std::vector<float> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < dim; ++j) {
                        float diff = a[j] - batch_flat[i * dim + j];
                        sum += diff * diff;
                    }
                    results[i] = std::sqrt(sum);
                }
                return results;
            },
            sizeof(float)));
    //--------------------------------------------------------------------------
    // barycentric_scores.comp - Simple pass through kernel
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "barycentric_scores", GPUFunction(
                                  {std::vector<float>{}}, 0, BARYCENTRIC_SCORES_COMP_ZST, 256, 1, 1,
                                  [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                                      const auto& in = params[0];
                                      return in;
                                  },
                                  sizeof(float)));

    //--------------------------------------------------------------------------
    // cross_entropy_loss.comp - Elementwise cross entropy
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "cross_entropy_loss",
        GPUFunction(
            {std::vector<float>{}, std::vector<float>{}}, 0, CROSS_ENTROPY_LOSS_COMP_ZST, 256, 1, 1,
            [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                const auto& pred = params[0];
                const auto& label = params[1];
                if (pred.size() != label.size())
                    throw std::runtime_error("cross_entropy_loss: size mismatch");
                std::vector<float> out(pred.size());
                for (size_t i = 0; i < pred.size(); ++i) {
                    float p = std::clamp(pred[i], 1e-7f, 1.0f - 1e-7f);
                    out[i] = -label[i] * std::log(p);
                }
                return out;
            },
            sizeof(float)));

    //--------------------------------------------------------------------------
    // fully_connected.comp - Basic dot product
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "fully_connected",
        GPUFunction(
            {std::vector<float>{}, std::vector<float>{}}, 0, FULLY_CONNECTED_COMP_ZST, 16, 16, 1,
            [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                const auto& w = params[0];
                const auto& x = params[1];
                if (w.size() != x.size())
                    throw std::runtime_error("fully_connected: size mismatch");
                float sum = 0.0f;
                for (size_t i = 0; i < w.size(); ++i)
                    sum += w[i] * x[i];
                std::vector<float> out(1, sum);
                return out;
            },
            sizeof(float)));

    //--------------------------------------------------------------------------
    // relu.comp - Elementwise ReLU
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "relu", GPUFunction(
                    {std::vector<float>{}}, 0, RELU_COMP_ZST, 256, 1, 1,
                    [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                        const auto& in = params[0];
                        std::vector<float> out(in.size());
                        for (size_t i = 0; i < in.size(); ++i)
                            out[i] = std::max(in[i], 0.0f);
                        return out;
                    },
                    sizeof(float)));

    //--------------------------------------------------------------------------
    // rmsprop.comp - RMSProp update
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "rmsprop", GPUFunction(
                       {std::vector<float>{}, std::vector<float>{}, std::vector<float>{}}, 0,
                       RMSPROP_COMP_ZST, 256, 1, 1,
                       [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                        auto param = params.at(0);
                           const auto& grad = params[1];
                           auto s = params[2];
                           if (param.size() != grad.size() || param.size() != s.size())
                               throw std::runtime_error("rmsprop: size mismatch");
                           float lr = 0.01f, decay = 0.9f, eps = 1e-8f;
                           for (size_t i = 0; i < param.size(); ++i) {
                               float g = grad[i];
                               s[i] = decay * s[i] + (1.0f - decay) * g * g;
                               param[i] -= lr * g / (std::sqrt(s[i]) + eps);
                           }
                           return param;
                       },
                       sizeof(float)));

    //--------------------------------------------------------------------------
    // sgd.comp - SGD update
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "sgd", GPUFunction(
                   {std::vector<float>{}, std::vector<float>{}}, 0, SGD_COMP_ZST, 256, 1, 1,
                   [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                       auto param = params[0];
                       const auto& grad = params[1];
                       if (param.size() != grad.size())
                           throw std::runtime_error("sgd: size mismatch");
                       float lr = 0.01f;
                       for (size_t i = 0; i < param.size(); ++i)
                           param[i] -= lr * grad[i];
                       return param;
                   },
                   sizeof(float)));

    //--------------------------------------------------------------------------
    // sigmoid.comp - Elementwise sigmoid
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "sigmoid", GPUFunction(
                       {std::vector<float>{}}, 0, SIGMOID_COMP_ZST, 256, 1, 1,
                       [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                           const auto& in = params[0];
                           std::vector<float> out(in.size());
                           for (size_t i = 0; i < in.size(); ++i)
                               out[i] = 1.0f / (1.0f + std::exp(-in[i]));
                           return out;
                       },
                       sizeof(float)));

    //--------------------------------------------------------------------------
    // adam.comp - Adam optimizer update
    //--------------------------------------------------------------------------
    GPUFunctionRegistry::getInstance().registerFunction(
        "adam", GPUFunction(
                    {std::vector<float>{}, std::vector<float>{}, std::vector<float>{},
                     std::vector<float>{}},
                    0, ADAM_COMP_ZST, 256, 1, 1,
                    [](const std::vector<std::vector<float>>& params) -> std::vector<float> {
                        auto param = params[0];
                        const auto& grad = params[1];
                        auto m = params[2];
                        auto v = params[3];
                        if (param.size() != grad.size() || param.size() != m.size() ||
                            param.size() != v.size())
                            throw std::runtime_error("adam: size mismatch");
                        float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
                        for (size_t i = 0; i < param.size(); ++i) {
                            m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
                            v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
                            float m_hat = m[i];
                            float v_hat = v[i];
                            param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                        }
                        return param;
                    },
                    sizeof(float)));
}

} // namespace psyne
