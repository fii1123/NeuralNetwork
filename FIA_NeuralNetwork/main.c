#include "FIA_NeuralNetwork.h"
#include <math.h> // Нужно для сигмоида

#define NN_UINT

//Информация о нейросети
void NN_Info(Neiro_Net *N_N)
{
    printf("Neiro network info:\nCount Layers: %d", N_N->Layers_count);
    NN_TYPE_C i, p, t;
    for (i = 0; i < N_N->Layers_count; ++i) {
        printf("\nLayer [%d] Neirones:%d ", i, N_N->Layers_NN[i].count);
        if(N_N->Layers_NN[i].bias != NULL){
            printf("Bias = %f", *N_N->Layers_NN[i].bias);
        }
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
            if(N_N->Layers_NN[i].N[p].weight != NULL){
                for (t = 0; t <  N_N->Layers_NN[i + 1].count; ++t) {
                     printf("\nWeight [%d]&[%d] = %f", p, t,
                            N_N->Layers_NN[i].N[p].weight[t]);
                }
            }else{
                printf("\nEnd\n");
            }
        }
    }
}

//функция активации (сигмоид)
NN_TYPE_D sigmoid(NN_TYPE_D a)
{
    return 1 / (1 + exp(-a));
}


int main(int argc, char *argv[])
{

//Загружаем структуру и веса нейросети из файла

puts("\n==================== Alpha =======================\n");

Neiro_Net *Alpha = NN_Load("../neironet.txt");

NN_Info(Alpha);

// Создаем массивы, где:
NN_TYPE_D IN[2] = {1.0, 0.0};                       // Входные данные (A^B)
NN_TYPE_D OUT[1];                                   // То, что вычисляет нейросеть
NN_TYPE_D Ideal_OUT[] = {1};                        //то, что должно получиться

//Первый запуск
NN_Clear(Alpha);                                    //очищаем веса
Result(Alpha, IN, OUT, sigmoid);                    //вычисляем первый результат

printf("Before Training result: %f\nError: %f", OUT[0],
        NN_Error_MSE(OUT, Ideal_OUT, 1));

//Первое обучение
NN_SetGlobalParam(Alpha, 0.03, 50);                 //Назначение гиперпараметров

NN_Backpropagation(Alpha, IN, Ideal_OUT, sigmoid);  //Обучение нейросети

Result(Alpha, IN, OUT, sigmoid);
printf("\n\nAfter Training result: %f\nError: %f", OUT[0],
        NN_Error_MSE(OUT, Ideal_OUT, 1));

NN_Save(Alpha, "../neironet2.txt");                 // сохраняем сеть в новый файл

NN_Delete(Alpha);                                   //удаление нейросети

puts("\n===================================================\n");


NN_TYPE_C laye[] = {2, 2, 1};

// Пример генерации сети с теми же весами

puts("\n==================== Beta ========================\n");

Neiro_Net *Beta = NN_Create(laye, 3, NULL);

// Первый нейрон
Beta->Layers_NN[0].N[0].weight[0] = 1.5;
Beta->Layers_NN[0].N[0].weight[1] = -2.3;

// Второй нейрон
Beta->Layers_NN[0].N[1].weight[0] = 0.45;
Beta->Layers_NN[0].N[1].weight[1] = 0.78;

// Последний
Beta->Layers_NN[1].N[0].weight[0] = -0.12;
Beta->Layers_NN[1].N[1].weight[0] = 0.13;

NN_Info(Beta);

NN_Save(Beta, "../neironet.txt");

NN_Delete(Beta);

    return 0;
}
