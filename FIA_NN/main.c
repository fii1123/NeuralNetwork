#include "FIA_NN.h"
#include "NN_func.h"


//Информация о нейросети
void NN_Info(Neuron_Net *N_N)
{
    printf("Neuron network info:\nCount Layers: %d", N_N->Layers_count);
    unsigned int i, p, t;
    for (i = 0; i < N_N->Layers_count; ++i) {
        printf("\nLayer [%d] Neurones:%d ", i, N_N->Layers_NN[i].count);
        if(N_N->Layers_NN[i].bias != NULL){
            printf("Bias");
            for (t = 0; t <  N_N->Layers_NN[i + 1].count; ++t) {
                 printf("\n /tWeight [%d] = %lf", t,
                        N_N->Layers_NN[i].bias->weight[t]);
            }
        }
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
            if(N_N->Layers_NN[i].N[p].weight != NULL){
                for (t = 0; t <  N_N->Layers_NN[i + 1].count; ++t) {
                     printf("\nWeight [%d]&[%d] = %3.6lf", p, t,
                            N_N->Layers_NN[i].N[p].weight[t]);
                }
            }
            else{
                printf("\nEnd\n");
            }
        }
    }
}


int main(int argc, char *argv[])
{

    // Создаем массивы, где:
    double IN[2] = {1.0, 0.0};          // Входные данные (A^B)
    double OUT[1];                      // То, что вычисляет нейросеть
    double Ideal_OUT[] = {1};           //то, что должно получиться


puts("\n==================== Alpha =======================\n");

Neuron_Net *Alpha = NN_Load("../Neuronnet.txt");

if (Alpha != NULL) {

    NN_Info(Alpha);

    //Первый запуск

    Result(Alpha, IN, OUT, sigmoid);            //вычисляем первый результат

    printf("Before Training result: %lf\nError: %lf\n\n", OUT[0],
            Error_MSE(OUT, Ideal_OUT, 1));

    Alpha->Alpha = 0.03;
    Alpha->Epsilon = 50;

    NN_Backpropagation(Alpha, IN, Ideal_OUT, sigmoid);  //Обучение нейросети

    NN_Info(Alpha);

    Result(Alpha, IN, OUT, sigmoid);
    printf("\n\nAfter Training result: %lf\nError: %lf", OUT[0],
            Error_MSE(OUT, Ideal_OUT, 1));

    NN_Save(Alpha, "../Neuronnet2.txt");        // сохраняем сеть в новый файл

    NN_Delete(Alpha);                           //удаление нейросети
}


puts("\n===================================================\n");

puts("\n==================== Beta ========================\n");
unsigned int laye[] = {2, 2, 1};

// Пример генерации сети с теми же весами

Neuron_Net *Beta = NN_Create(laye, 3, NULL);

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

Result(Beta, IN, OUT, sigmoid);                    //вычисляем первый результат

printf("Before Training result: %lf\nError: %lf\n\n", OUT[0],
        Error_MSE(OUT, Ideal_OUT, 1));


Beta->Alpha = 0.03;
Beta->Epsilon = 50;


NN_Backpropagation(Beta, IN, Ideal_OUT, sigmoid);  //Обучение нейросети

NN_Info(Beta);

Result(Beta, IN, OUT, sigmoid);
printf("\n\nAfter Training result: %lf\nError: %lf", OUT[0],
        Error_MSE(OUT, Ideal_OUT, 1));


NN_Save(Beta, "../Neuronnet3.txt");

NN_Delete(Beta);

puts("\n===================================================\n");

getchar();
    return 0;
}
