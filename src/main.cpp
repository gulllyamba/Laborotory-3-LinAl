#include <iostream>
#include "Qubit.hpp"
int main() {
    std::cout << "1) Функция f(x1, x2) = 1 :\n";
    MatrixXcd One(4, 4);
    One << -1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            0.0, 0.0, 0.0, -1.0;
    Condition Cond_1(2);
    Gate Oracle_1(One);
    Cond_1.Iteration(Oracle_1);
    std::cout << Cond_1 << "\n";
    std::cout << "Вероятность правильного ответа: " << Cond_1.CalculateProbability(Oracle_1) << "\n\n";

    std::cout << "2) Функция f(x1, x2) = x1 || x2 :\n";
    MatrixXcd Or(4, 4);
    Or << 1.0, 0.0, 0.0, 0.0,
          0.0, -1.0, 0.0, 0.0,
          0.0, 0.0, -1.0, 0.0,
          0.0, 0.0, 0.0, -1.0;
    Condition Cond_2(2);
    Gate Oracle_2(Or);
    Cond_2.Iteration(Oracle_2);
    std::cout << Cond_2 << "\n";
    std::cout << "Вероятность правильного ответа: " << Cond_2.CalculateProbability(Oracle_2) << "\n\n";
    
    std::cout << "3) Функция f(x1, x2) = x1 && x2 :\n";
    MatrixXcd And(4, 4);
    And << 1.0, 0.0, 0.0, 0.0,
           0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 1.0, 0.0,
           0.0, 0.0, 0.0, -1.0;
    Condition Cond_3(2);
    Gate Oracle_3(And);
    Cond_3.Iteration(Oracle_3);
    std::cout << Cond_3 << "\n";
    std::cout << "Вероятность правильного ответа :" << Cond_3.CalculateProbability(Oracle_3) << "\n\n";

    std::cout << "4) Функция f(x1, x2) = x1 XOR x2 :\n";
    MatrixXcd Xor(4, 4);
    Xor << 1.0, 0.0, 0.0, 0.0,
           0.0, -1.0, 0.0, 0.0,
           0.0, 0.0, -1.0, 0.0,
           0.0, 0.0, 0.0, 1.0;
    Condition Cond_4(2);
    Gate Oracle_4(Xor);
    Cond_4.Iteration(Oracle_4);
    std::cout << Cond_4 << "\n";
    std::cout << "Вероятность правильного ответа: " << Cond_4.CalculateProbability(Oracle_4) << "\n\n";

    std::cout << "5) Некая функция f(x1, x2, x3) :\n";
    MatrixXcd func_3(8, 8);
    func_3 << -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Condition Cond_5(3);
    Gate Oracle_5(func_3);
    Cond_5.Iteration(Oracle_5);
    std::cout << Cond_5 << "\n";
    std::cout <<"Вероятность правильного ответа: " << Cond_5.CalculateProbability(Oracle_5) << "\n\n";

    std::cout << "6) Некая функция f(x1, x2, x3, x4):\n";
    MatrixXcd func_4(16, 16);
    func_4 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Condition Cond_6(4);
    Gate Oracle_6(func_4);
    Cond_6.Iteration(Oracle_6);
    std::cout << Cond_6 << "\n";
    std::cout << "Вероятность правильного ответа: " << Cond_6.CalculateProbability(Oracle_6) << "\n\n";
}