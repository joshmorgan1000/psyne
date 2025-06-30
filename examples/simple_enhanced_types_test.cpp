#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;
using namespace psyne::types;

int main() {
    try {
        std::cout << "Testing Enhanced Message Types" << std::endl;
        std::cout << "==============================" << std::endl;

        // Create a channel
        auto channel = create_channel("memory://enhanced_test", 1024 * 1024);

        // Test Matrix4x4f
        std::cout << "\nTesting Matrix4x4f..." << std::endl;
        Matrix4x4f matrix(*channel);
        matrix.initialize();
        matrix(0, 1) = 5.0f;
        matrix(1, 0) = 10.0f;

        std::cout << "Matrix element (0,1): " << matrix(0, 1) << std::endl;
        std::cout << "Matrix element (1,0): " << matrix(1, 0) << std::endl;
        std::cout << "Matrix trace: " << matrix.trace() << std::endl;

        // Test Vector3f
        std::cout << "\nTesting Vector3f..." << std::endl;
        Vector3f vec(*channel);
        vec.initialize();
        vec.x() = 3.0f;
        vec.y() = 4.0f;
        vec.z() = 0.0f;

        std::cout << "Vector: (" << vec.x() << ", " << vec.y() << ", "
                  << vec.z() << ")" << std::endl;
        std::cout << "Vector length: " << vec.length() << std::endl;

        vec *= 2.0f;
        std::cout << "After scaling by 2: (" << vec.x() << ", " << vec.y()
                  << ", " << vec.z() << ")" << std::endl;
        std::cout << "New length: " << vec.length() << std::endl;

        // Test Int8Vector
        std::cout << "\nTesting Int8Vector..." << std::endl;
        Int8Vector qvec(*channel);
        qvec.initialize();
        qvec.resize(5);
        qvec.set_quantization_params(0.1f, 128);

        for (size_t i = 0; i < qvec.size(); ++i) {
            qvec[i] = static_cast<int8_t>(i * 10);
        }

        std::cout << "Quantized vector size: " << qvec.size() << std::endl;
        std::cout << "Scale: " << qvec.header().scale
                  << ", Zero point: " << qvec.header().zero_point << std::endl;
        std::cout << "Values: ";
        for (size_t i = 0; i < qvec.size(); ++i) {
            std::cout << (int)qvec[i] << " ";
        }
        std::cout << std::endl;

        // Test ComplexVectorF
        std::cout << "\nTesting ComplexVectorF..." << std::endl;
        ComplexVectorF cvec(*channel);
        cvec.initialize();
        cvec.resize(3);

        cvec[0] = std::complex<float>(1.0f, 0.0f);
        cvec[1] = std::complex<float>(0.0f, 1.0f);
        cvec[2] = std::complex<float>(1.0f, 1.0f);

        std::cout << "Complex vector size: " << cvec.size() << std::endl;
        std::cout << "Total power: " << cvec.power() << std::endl;

        cvec.conjugate();
        std::cout << "After conjugation: ";
        for (size_t i = 0; i < cvec.size(); ++i) {
            std::cout << "(" << cvec[i].real() << ", " << cvec[i].imag()
                      << ") ";
        }
        std::cout << std::endl;

        // Test MLTensorF
        std::cout << "\nTesting MLTensorF..." << std::endl;
        MLTensorF tensor(*channel);
        tensor.initialize();
        tensor.set_shape({2, 3, 4}, MLTensorF::Layout::NCHW);

        std::cout << "Tensor dimensions: ";
        auto shape = tensor.shape();
        for (size_t dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Total elements: " << tensor.total_elements() << std::endl;

        // Test SparseMatrixF
        std::cout << "\nTesting SparseMatrixF..." << std::endl;
        SparseMatrixF sparse(*channel);
        sparse.initialize();
        sparse.set_structure(3, 3, 3); // 3x3 matrix with 3 non-zero elements

        std::cout << "Sparse matrix: " << sparse.rows() << "x" << sparse.cols()
                  << " with " << sparse.nnz() << " non-zero elements"
                  << std::endl;

        std::cout << "\n✅ All enhanced message types working!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}