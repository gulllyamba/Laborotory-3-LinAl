#ifndef QUBIT_HPP
#define QUBIT_HPP

#include <iostream>
#include <math.h>
#include <../eigen3/Eigen/Dense>

using namespace Eigen;
using Complex = std::complex<double>;

class Condition;

class Gate {
    public:
        Gate(MatrixXcd matrix) :gate(matrix) {}
        Gate(const Gate& other) : gate(other.gate) {}

        friend Condition operator*(const Gate& g, const Condition& cond);
        Gate operator*(const Gate& other) {
            int rows = gate.rows() * other.gate.rows();
            int cols = gate.cols() * other.gate.cols();
            MatrixXcd result(rows, cols);
            for (int i = 0; i < gate.rows(); i++) {
                for (int j = 0; j < gate.cols(); j++) {
                    result.block(i * other.gate.rows(), j * other.gate.cols(), other.gate.rows(), other.gate.cols()) = gate(i, j) * other.gate;
                }
            }
            return Gate(result);
        }

        friend std::ostream& operator<<(std::ostream& os, const Gate& other) {
            os << other.gate;
            return os;
        }

        Gate& operator=(const Gate& other) {
            if (this != &other) {
                gate = other.gate;
            }
            return *this;
        }

        static Gate PauliX() {
            Matrix2cd X;
            X << 0.0, 1.0,
                 1.0, 0.0;
            return Gate(X);
        }

        static Gate PauliY() {
            Matrix2cd Y;
            Y << 0.0, Complex(0.0, -1.0),
                 Complex(0.0, 1.0), 0.0;
            return Gate(Y);
        }

        static Gate PauliZ() {
            Matrix2cd Z;
            Z << 1.0, 0.0,
                 0.0, -1.0;
            return Gate(Z);
        }

        static Gate Hadamard() {
            double inv_sqrt2 = 1.0 / sqrt(2.0);
            Matrix2cd H;
            H << inv_sqrt2, inv_sqrt2,
                 inv_sqrt2, -inv_sqrt2;
            return Gate(H);
        }

        static Gate CNOT() {
            MatrixXcd CNOT;
            CNOT << 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 0.0;
            return Gate(CNOT);
        }

    private:
        MatrixXcd gate;
        friend class Condition;
};

class Condition {
    public:
        Condition(int cnt) : ket(VectorXcd::Zero(1 << cnt)) {
            ket(0) = 1.0;
        }
        Condition(VectorXcd cond){
            ket = cond.normalized();
        }

        friend Condition operator*(const Gate& g, const Condition& cond);
        Condition operator*(const Condition& cond) {
            return Condition(ket * (cond.ket.transpose()));
        }

        friend std::ostream& operator<<(std::ostream& os, const Condition& cond) {
            os << cond.ket;
            return os;
        }

        Gate DiffGate() {
            int dim = ket.rows();
            VectorXcd phi0 = VectorXcd::Constant(dim, 1.0 / std::sqrt(dim)); 
            MatrixXcd D = 2.0 * phi0 * phi0.adjoint() - MatrixXcd::Identity(dim, dim); 
            return Gate(D);
        }

        double CalculateProbability(const Gate& Oracle) const {
            double probability = 0.0;
            for (int i = 0; i < Oracle.gate.rows(); i++) {
                if (std::abs(Oracle.gate(i, i).real() - (-1.0)) < 1e-9) {
                    probability += std::norm(ket(i));
                }
            }
            return probability;
        }

        Condition Iteration(const Gate& Oracle) {
            int cnt = 0;
            for (int i = 0; i < Oracle.gate.rows(); i++) {
                if (std::abs(Oracle.gate(i, i).real() + 1.0) < 1e-9) cnt++;
            }
            if (cnt == 0) {
                throw std::runtime_error("Ни один элемент из множества не удовлетворяет оракулу");
            }
            Gate H = Gate::Hadamard();
            for (int i = 0; i < std::log2(ket.rows()) - 1; i++) {
                H = H * Gate::Hadamard();
            }
            *this = H * (*this);
            cnt = sqrt(ket.rows() / cnt);
            Gate Diff = DiffGate();
            int i = 1;
            do{
                *this = Oracle * (*this);
                *this = Diff * (*this);
                i++;
            } while (CalculateProbability(Oracle) < 90 && i < cnt);
            return *this;
        }

    private:
	    VectorXcd ket;
        friend class Gate;
};

Condition operator*(const Gate& g, const Condition& cond) {
	return Condition(g.gate * cond.ket);
}

#endif // QUBIT_HPP