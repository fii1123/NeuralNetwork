#ifndef FIA_NEURALNETWORK_H
#define FIA_NEURALNETWORK_H

#include <stdio.h>
#include <malloc.h>

//Структура нейрона
struct Neuron
{
    double data;           //Данные
    double *weight;        //Массив веса связи с прочими нейронами
        //Для обучения
    double delta;
    double *last_weight;   //Массив предыдущей эпохи
};

//Структура слоя
struct Layer
{
    unsigned int count;     //Число нейронов на слое
    struct Neuron *bias;    // нейрон смещения
    struct Neuron *N;       //Массив нейронов
};

//Нейросеть
typedef struct
{
    struct Layer *Layers_NN;        //Массив слоёв
    unsigned int Layers_count;      //Число слоёв в нейросети
    unsigned int Input_count;       //Число входных нейронов
    unsigned int Output_count;      //Число выходных нейронов

    //Гиперпараметры
    double Epsilon;     // Скорость обучения
    double Alpha;       // Момент
} Neuron_Net;

//Инициализация нейросети из заданных массивами параметров
Neuron_Net* NN_Create(unsigned int *Layers, unsigned int LayersCount,
                      int *Biases);

//Инициализация нейросети из файла
Neuron_Net* NN_Load(char *file_path);

//Сохранение нейросети
void NN_Save(Neuron_Net *N_N, char* file_path);

//Отчистка нейронов от данных
void NN_Clear(Neuron_Net *N_N);

//Функция вычисления
void Result(Neuron_Net *N_N, double *input, double *output,
            double activ(double x));

//Обучение
void NN_Backpropagation (Neuron_Net *N_N, double *input, double *output,
                          double activ(double x));
//Удаление нейросети
void NN_Delete(Neuron_Net *N_N);


#endif // FIA_NEURALNETWORK_H
