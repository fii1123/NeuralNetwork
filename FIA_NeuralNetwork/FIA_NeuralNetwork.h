#ifndef FIA_NEURALNETWORK_H
#define FIA_NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>

#include <time.h> // Нужно для рандома

//Типы данных для нейросети
// #define NN_UINT // переключает количества с unsigned short на unsigned int
#ifdef NN_UINT
#define NN_TYPE_C unsigned int
#else
#define NN_TYPE_C unsigned short
#endif
// #define NN_DATA_DOUBLE // переключает веса и данные с float на double
#ifdef NN_DATA_DOUBLE
#define NN_TYPE_D double
#define NN_FILE_R "%lf"
#else
#define NN_TYPE_D float
#define NN_FILE_R "%f"
#endif

#define RANDOM_WEIGDH (NN_TYPE_D) ((rand() + 1000) / 1000)

//Структура нейрона
struct Neiro
{
    NN_TYPE_D data;           //Данные
    NN_TYPE_D *weight;        //Массив веса связи с прочими нейронами
    //Для обучения
    NN_TYPE_D delta;
    NN_TYPE_D *last_weight;   //Массив предыдущей эпохи
};
//Структура слоя
struct Layer
{
    NN_TYPE_C count;    //Число нейронов на слое
    NN_TYPE_D *bias;    //Массив веса нейрона смещения (NULL, если отсутствует)
    struct Neiro *N;    //Массив нейронов
};
//Нейросеть
typedef struct
{
    struct Layer *Layers_NN;    //Массив слоёв
    NN_TYPE_C Layers_count;     //Число слоёв в нейросети
    NN_TYPE_C Input_count;      //Число входных нейронов
    NN_TYPE_C Output_count;     //Число выходных нейронов

    //Гиперпараметры
    NN_TYPE_D Epsilon;          // Скорость обучения
    NN_TYPE_D Alpha;            // Момент
} Neiro_Net;


Neiro_Net* NN_Create(NN_TYPE_C *Layers, NN_TYPE_C LayersCount, int *Biases);

//Инициализация нейросети из файла
Neiro_Net* NN_Load(char *file_path);

//Сохранение нейросети
void NN_Save(Neiro_Net *N_N, char* file_path);

//Отчистка нейросети (данных, но не веса)
void NN_Clear(Neiro_Net *N_N);

//Функция вычисления
void Result(Neiro_Net *N_N, NN_TYPE_D *input, NN_TYPE_D *output, NN_TYPE_D activ(NN_TYPE_D x));

//Вычисление ошибки MSE
NN_TYPE_D NN_Error_MSE(NN_TYPE_D* output_NN, NN_TYPE_D *output_Ideal,
                       const NN_TYPE_C size);
//Настройка гиперпараметров
void NN_SetGlobalParam(Neiro_Net *N_N, NN_TYPE_D Alpha, NN_TYPE_D Epsilon);

//Обучение
void NN_Backpropagation (Neiro_Net *N_N, NN_TYPE_D *input, NN_TYPE_D *output,
                          NN_TYPE_D activ(NN_TYPE_D x));
//Удаление нейросети
void NN_Delete(Neiro_Net *N_N);


#endif // FIA_NEURALNETWORK_H
