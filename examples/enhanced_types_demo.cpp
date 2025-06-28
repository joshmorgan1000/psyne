// Enhanced Message Types Demo - showcases all new message types in psyne
// Demonstrates fixed matrices, quantized vectors, complex numbers, ML tensors,
// and sparse matrices

#include <algorithm>
#include <complex>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <vector>

using namespace psyne;
using namespace psyne::types;

void demo_fixed_matrices() {
    std::cout << "\n=== Fixed Matrix Types Demo ===\n";

    auto channel = create_channel("memory://matrices", 1024 * 1024);

    // Create a 4x4 transformation matrix
    Matrix4x4f transform(*channel);
    transform.initialize();

    // Set up a simple transformation matrix
    auto eigen_mat = transform.as_eigen();
    eigen_mat.setIdentity();
    eigen_mat(0, 3) = 10.0f; // Translation in X
    eigen_mat(1, 3) = 20.0f; // Translation in Y
    eigen_mat(2, 3) = 30.0f; // Translation in Z

    std::cout << "Transform matrix determinant: " << transform.determinant()
              << std::endl;
    std::cout << "Transform matrix trace: " << transform.trace() << std::endl;

    // Create a 3D vector
    Vector3f position(*channel);
    position.initialize();
    position.x() = 1.0f;
    position.y() = 2.0f;
    position.z() = 3.0f;

    std::cout << "Position vector length: " << position.length() << std::endl;

    // In-place operations
    position *= 2.0f;
    std::cout << "Scaled position vector length: " << position.length()
              << std::endl;

    // Demonstrate sending/receiving
    transform.send();
    position.send();

    auto received_transform = channel->receive<Matrix4x4f>();
    auto received_position = channel->receive<Vector3f>();

    if (received_transform && received_position) {
        std::cout << "Successfully sent and received matrices!\n";
        std::cout << "Received transform determinant: "
                  << received_transform->determinant() << std::endl;
        std::cout << "Received position length: " << received_position->length()
                  << std::endl;
    }
}

void demo_quantized_vectors() {
    std::cout << "\n=== Quantized Vector Types Demo ===\n";

    auto channel = create_channel("memory://quantized", 1024 * 1024);

    // Original float data
    std::vector<float> float_data = {0.1f,  0.5f, -0.3f, 0.8f,
                                     -0.9f, 0.2f, 0.7f,  -0.1f};

    // Quantize to int8
    Int8Vector quantized(*channel);
    quantized.initialize();

    float scale = 0.01f;
    int8_t zero_point = 0;
    quantized.quantize_from_float(float_data.data(), float_data.size(), scale,
                                  zero_point);

    std::cout << "Original float data size: "
              << float_data.size() * sizeof(float) << " bytes\n";
    std::cout << "Quantized int8 data size: "
              << quantized.size() * sizeof(int8_t) << " bytes\n";
    std::cout << "Compression ratio: "
              << (float_data.size() * sizeof(float)) /
                     (float)(quantized.size() * sizeof(int8_t))
              << "x\n";

    std::cout << "Min quantized value: " << (int)quantized.min_value()
              << std::endl;
    std::cout << "Max quantized value: " << (int)quantized.max_value()
              << std::endl;
    std::cout << "Sum of quantized values: " << quantized.sum() << std::endl;

    // Dequantize back to float
    std::vector<float> dequantized(quantized.size());
    quantized.dequantize_to_float(dequantized.data(), scale, zero_point);

    std::cout << "Dequantization error (first few elements):\n";
    for (size_t i = 0; i < std::min(size_t(4), float_data.size()); ++i) {
        float error = std::abs(float_data[i] - dequantized[i]);
        std::cout << "  [" << i << "] Original: " << float_data[i]
                  << ", Dequantized: " << dequantized[i] << ", Error: " << error
                  << std::endl;
    }
}

void demo_complex_vectors() {
    std::cout << "\n=== Complex Vector Types Demo ===\n";

    auto channel = create_channel("memory://complex", 1024 * 1024);

    ComplexVectorF signal(*channel);
    signal.initialize();

    // Create a complex sinusoid signal
    signal = {
        {1.0f, 0.0f},       // 1 + 0i
        {0.707f, 0.707f},   // e^(iπ/4)
        {0.0f, 1.0f},       // i
        {-0.707f, 0.707f},  // e^(i3π/4)
        {-1.0f, 0.0f},      // -1
        {-0.707f, -0.707f}, // e^(i5π/4)
        {0.0f, -1.0f},      // -i
        {0.707f, -0.707f}   // e^(i7π/4)
    };

    std::cout << "Complex signal power: " << signal.power() << std::endl;

    // Get magnitude and phase
    auto magnitudes = signal.magnitude();
    auto phases = signal.phase();

    std::cout << "First few magnitude/phase pairs:\n";
    for (size_t i = 0; i < std::min(size_t(4), magnitudes.size()); ++i) {
        std::cout << "  [" << i << "] Magnitude: " << magnitudes[i]
                  << ", Phase: " << phases[i] << " radians" << std::endl;
    }

    // Complex conjugate
    signal.conjugate();
    std::cout << "After conjugation, power: " << signal.power() << std::endl;

    // Create another signal for dot product
    ComplexVectorF signal2(*channel);
    signal2.initialize();
    signal2 = {{1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}};
    signal2.resize(signal.size()); // Match sizes

    auto dot_result = signal.dot_product(signal2);
    std::cout << "Dot product result: " << dot_result.real() << " + "
              << dot_result.imag() << "i" << std::endl;
}

void demo_ml_tensors() {
    std::cout << "\n=== ML Tensor Demo ===\n";

    auto channel = create_channel("memory://ml",
                                  2 * 1024 * 1024); // Larger buffer for tensors

    MLTensorF image_batch(*channel);
    image_batch.initialize();

    // Set up a batch of images: 1 image, 3 channels, 8x8 pixels (NCHW format)
    // Small size to fit in default message buffer
    image_batch.set_shape({1, 3, 8, 8}, MLTensorF::Layout::NCHW);

    std::cout << "Tensor shape: ";
    auto shape = image_batch.shape();
    for (size_t dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "Total elements: " << image_batch.total_elements()
              << std::endl;
    std::cout << "Batch size: " << image_batch.batch_size() << std::endl;

    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < image_batch.total_elements(); ++i) {
        image_batch[i] = dis(gen);
    }

    std::cout << "Initial stats - Min: " << image_batch.min()
              << ", Max: " << image_batch.max()
              << ", Mean: " << image_batch.mean() << std::endl;

    // Apply ReLU activation
    image_batch.relu();
    std::cout << "After ReLU - Min: " << image_batch.min()
              << ", Max: " << image_batch.max()
              << ", Mean: " << image_batch.mean() << std::endl;

    // Access specific elements using NCHW indexing
    float pixel_value =
        image_batch.at_nchw(0, 1, 4, 4); // Batch 0, Channel 1, Y=4, X=4
    std::cout << "Pixel at (0,1,4,4): " << pixel_value << std::endl;

    // Normalize using L2 norm
    image_batch.normalize_l2();
    std::cout << "After L2 normalization - Mean: " << image_batch.mean()
              << std::endl;
}

void demo_sparse_matrices() {
    std::cout << "\n=== Sparse Matrix Demo ===\n";

    auto channel = create_channel("memory://sparse", 1024 * 1024);

    SparseMatrixF sparse(*channel);
    sparse.initialize();
    sparse.set_dimensions(100, 100);
    sparse.reserve_nnz(500); // Reserve space for 500 non-zeros

    // Create a sparse diagonal matrix with some random off-diagonal elements
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.1f, 1.0f);
    std::uniform_int_distribution<int> coord_dis(0, 99);

    // Add diagonal elements
    for (size_t i = 0; i < 100; ++i) {
        sparse.set(i, i, dis(gen));
    }

    // Add some random off-diagonal elements
    for (int i = 0; i < 50; ++i) {
        size_t row = coord_dis(gen);
        size_t col = coord_dis(gen);
        if (row != col) { // Avoid overwriting diagonal
            sparse.set(row, col,
                       dis(gen) * 0.1f); // Smaller off-diagonal values
        }
    }

    std::cout << "Sparse matrix dimensions: " << sparse.rows() << "x"
              << sparse.cols() << std::endl;
    std::cout << "Number of non-zeros: " << sparse.nnz() << std::endl;
    std::cout << "Sparsity: "
              << (1.0 -
                  (double)sparse.nnz() / (sparse.rows() * sparse.cols())) *
                     100.0
              << "%" << std::endl;
    std::cout << "Frobenius norm: " << sparse.frobenius_norm() << std::endl;
    std::cout << "Trace: " << sparse.trace() << std::endl;

    // Test sparse matrix-vector multiplication
    std::vector<float> x(100, 1.0f); // Vector of ones
    std::vector<float> y(100);

    sparse.matvec(x.data(), y.data());

    // The result should be close to the row sums
    float max_y = *std::max_element(y.begin(), y.end());
    float min_y = *std::min_element(y.begin(), y.end());
    std::cout << "Matrix-vector product range: [" << min_y << ", " << max_y
              << "]" << std::endl;

    // Test row access
    auto row_view = sparse.row(0);
    std::cout << "Row 0 has " << row_view.size
              << " non-zero elements:" << std::endl;
    size_t count = 0;
    for (auto [col_idx, value] : row_view) {
        if (count++ < 5) { // Show first 5 elements
            std::cout << "  Column " << col_idx << ": " << value << std::endl;
        }
    }
    if (row_view.size > 5) {
        std::cout << "  ... and " << (row_view.size - 5) << " more"
                  << std::endl;
    }
}

int main() {
    std::cout << "Psyne Enhanced Message Types Demo\n";
    std::cout << "==================================\n";

    try {
        demo_fixed_matrices();
        demo_quantized_vectors();
        demo_complex_vectors();
        demo_ml_tensors();
        demo_sparse_matrices();

        std::cout << "\n=== Demo Complete ===\n";
        std::cout << "All enhanced message types demonstrated successfully!\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
