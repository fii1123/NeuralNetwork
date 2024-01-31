#ifndef NN_FUNC_H
#define NN_FUNC_H

#include <math.h> // Нужно для сигмоида

//функция активации (сигмоид)
double sigmoid(double a)
{
    return (double) 1 / (1 + exp(-a));
}

//Вычисление среднеквадратичной ошибки MSE
double Error_MSE(double *output_NN, double *output_Ideal,
                       const unsigned int size)
{
    unsigned int i;
    double a = 0, res = 0;
    for(i = 0; i < size; i++){
        a = output_Ideal[i] - output_NN[i];
        res += (a * a);
    }
    return res / size;
}

#endif // NN_FUNC_H
