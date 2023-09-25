#include "FIA_NeuralNetwork.h"

Neiro_Net* NN_Create(NN_TYPE_C *Layers, NN_TYPE_C LayersCount, int *Biases)
{
    srand(time(NULL));
    NN_TYPE_C i, p, t;         //Переменные
    NN_TYPE_C indx_out = LayersCount - 1;       // Индекс последнего слоя (выхода)
    Neiro_Net *N_N = malloc(sizeof (Neiro_Net));

    N_N->Alpha = 0.0;
    N_N->Epsilon = 0.0;

    //Cчитывание числа слоёв и выходных нейронов для оптимизации
    N_N->Layers_count = LayersCount;
    N_N->Output_count = Layers[indx_out];

    //Создание слоёв
    N_N->Layers_NN = malloc((N_N->Layers_count) * sizeof(struct Layer));

    i = indx_out;

    //Обнуление последнего слоя
    N_N->Layers_NN[i].N = malloc((N_N->Output_count) * sizeof(struct Layer));
    N_N->Layers_NN[i].bias = NULL;
    N_N->Layers_NN[i].count = N_N->Output_count;
    for (p = 0; p < N_N->Output_count; ++p) {
        N_N->Layers_NN[i].N[p].weight = 0;
    }

    //Проход в обратном порядке
    --i;
    do {
        //Считывание числа нейронов на слое
        N_N->Layers_NN[i].count = Layers[i];

        //Если есть нейрон смещения
        if(Biases != NULL && Biases[i] != 0) {
            //На последнем слое
            if(i != indx_out - 1) {
                N_N->Layers_NN[i].bias = malloc(N_N->Layers_NN[i + 1].count
                        * sizeof(NN_TYPE_D));
                for (p = 0; p < N_N->Layers_NN[i + 1].count; ++p) {
                    N_N->Layers_NN[i].bias[p] = RANDOM_WEIGDH;
                }
            }
            //В остальных слоях
            else {
                N_N->Layers_NN[i].bias = malloc(N_N->Output_count
                                               * sizeof(NN_TYPE_D));
                for (p = 0; p < N_N->Output_count; ++p){
                    N_N->Layers_NN[i].bias[p] = RANDOM_WEIGDH;
                }
            }
        }
        else {
            N_N->Layers_NN[i].bias = NULL;
        }

        //Выделение памяти под нейроны
        N_N->Layers_NN[i].N = malloc(N_N->Layers_NN[i].count
                                   * sizeof(struct Neiro));
        //Проход по нейронам
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
            N_N->Layers_NN[i].N[p].data = 0.0;   //Обнуление значений

            //Если слои не граничат с выходом
            if(i != indx_out - 1){

                // Инициализация весов по числу нейронов предыдущего слоя
                N_N->Layers_NN[i].N[p].weight =
                        malloc(N_N->Layers_NN[i - 1].count * sizeof (NN_TYPE_D));
                N_N->Layers_NN[i].N[p].last_weight =
                        malloc(N_N->Layers_NN[i - 1].count * sizeof (NN_TYPE_D));

                //Заполнение весов
                for (t = 0; t < N_N->Layers_NN[i + 1].count; ++t) {
                    N_N->Layers_NN[i].N[p].weight[t] = RANDOM_WEIGDH;
                    N_N->Layers_NN[i].N[p].last_weight[t] = 0.0;
                }
            }
            //Если слой граничит с выходным
            else {

                N_N->Layers_NN[i].N[p].weight =
                        malloc(N_N->Output_count * sizeof(NN_TYPE_D));
                N_N->Layers_NN[i].N[p].last_weight =
                        malloc(N_N->Layers_NN[i - 1].count * sizeof (NN_TYPE_D));

                for (t = 0; t < N_N->Output_count; ++t) {
                    N_N->Layers_NN[i].N[p].weight[t] = RANDOM_WEIGDH;
                    N_N->Layers_NN[i].N[p].last_weight[t] = 0.0;
                }
            }
        }
        --i;
    } while(i == 0);

    N_N->Input_count = N_N->Layers_NN[0].count;

    return N_N;
}

Neiro_Net* NN_Load(char *file_path)
{
    FILE *neirofile = fopen(file_path, "rb");
    if(neirofile == NULL) {
        puts("Error of open file!");
    }
    else {
        NN_TYPE_C i, p, t;//Переменные
        Neiro_Net *N_N = malloc(sizeof (Neiro_Net));

        N_N->Alpha = 0.0;
        N_N->Epsilon = 0.0;

        //Cчитывание числа слоёв и выходных нейронов для оптимизации
        fread(&N_N->Layers_count, sizeof (NN_TYPE_C), 1,  neirofile);
        fread(&N_N->Output_count, sizeof (NN_TYPE_C), 1,  neirofile);

        NN_TYPE_C indx_out = N_N->Layers_count - 1;       // Индекс последнего слоя (выхода)

        //Создание слоёв
        N_N->Layers_NN = malloc((N_N->Layers_count) * sizeof(struct Layer));

        i = indx_out;

        //Обнуление последнего слоя
        N_N->Layers_NN[i].N = malloc((N_N->Output_count) * sizeof(struct Layer));
        N_N->Layers_NN[i].bias = NULL;
        N_N->Layers_NN[i].count = N_N->Output_count;
        for (p = 0; p < N_N->Output_count; ++p) {
            N_N->Layers_NN[i].N[p].weight = 0;
        }

        //Проход в обратном порядке
        --i;
        do {
            //Считывание числа нейронов на слое
            fread(&N_N->Layers_NN[i].count, sizeof (NN_TYPE_C), 1,  neirofile);

            //Если есть нейрон смещения
            char bias;
            fread(&bias, sizeof (char), 1,  neirofile);
            if(bias == 'b') {
                //На последнем слое
                if(i != indx_out - 1) {
                    N_N->Layers_NN[i].bias = malloc(N_N->Layers_NN[i + 1].count
                            * sizeof(NN_TYPE_D));
                    for (p = 0; p < N_N->Layers_NN[i + 1].count; ++p) {
                        fread(&N_N->Layers_NN[i].bias[p],
                              sizeof (NN_TYPE_D), 1,  neirofile);
                    }
                }
                //В остальных слоях
                else {
                    N_N->Layers_NN[i].bias = malloc(N_N->Output_count
                                                   * sizeof(NN_TYPE_D));
                    for (p = 0; p < N_N->Output_count; ++p){
                        fread(&N_N->Layers_NN[i].bias[p],
                              sizeof (NN_TYPE_D), 1,  neirofile);
                    }
                }
            }
            else {
                N_N->Layers_NN[i].bias = NULL;
            }

            //Выделение памяти под нейроны
            N_N->Layers_NN[i].N = malloc(N_N->Layers_NN[i].count
                                       * sizeof(struct Neiro));
            //Проход по нейронам
            for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
                N_N->Layers_NN[i].N[p].data = 0.0;   //Обнуление значений

                //Если слои не граничат с выходом
                if(i != indx_out - 1){

                    // Инициализация весов по числу нейронов предыдущего слоя
                    N_N->Layers_NN[i].N[p].weight =
                            malloc(N_N->Layers_NN[i - 1].count * sizeof (NN_TYPE_D));
                    N_N->Layers_NN[i].N[p].last_weight =
                            malloc(N_N->Layers_NN[i - 1].count * sizeof (NN_TYPE_D));

                    //Заполнение весов
                    for (t = 0; t < N_N->Layers_NN[i + 1].count; ++t) {
                        fread(&N_N->Layers_NN[i].N[p].weight[t],
                              sizeof (NN_TYPE_D), 1,  neirofile);
                        N_N->Layers_NN[i].N[p].last_weight[t] = 0.0;
                    }
                }
                //Если слой граничит с выходным
                else {

                    N_N->Layers_NN[i].N[p].weight =
                            malloc(N_N->Output_count * sizeof(NN_TYPE_D));
                    N_N->Layers_NN[i].N[p].last_weight =
                            malloc(N_N->Layers_NN[i - 1].count * sizeof (NN_TYPE_D));

                    for (t = 0; t < N_N->Output_count; ++t) {
                        fread(&N_N->Layers_NN[i].N[p].weight[t],
                              sizeof (NN_TYPE_D), 1,  neirofile);
                        N_N->Layers_NN[i].N[p].last_weight[t] = 0.0;
                    }
                }
            }
            --i;
        } while(i == 0);

        N_N->Input_count = N_N->Layers_NN[0].count;
        fclose(neirofile);
        return N_N;
    }
    return NULL;
}

//Удаление нейросети
void NN_Delete(Neiro_Net *N_N)
{
    free(N_N->Layers_NN);
}

//Сохранение нейросети
void NN_Save(Neiro_Net *N_N, char* file_path)
{
    NN_TYPE_C i, p, t;//Переменные
    FILE *neirofile = fopen(file_path, "wb");

    fwrite(&N_N->Layers_count, sizeof (NN_TYPE_C), 1,  neirofile);
    fwrite(&N_N->Output_count, sizeof (NN_TYPE_C), 1,  neirofile);

    i = N_N->Layers_count - 2;
    do{
        //Запись числа нейронов на слое
        fwrite(&N_N->Layers_NN[i].count, sizeof (NN_TYPE_C), 1,  neirofile);

        //Если есть нейрон смещения
        if(N_N->Layers_NN[i].bias != NULL) {
            fwrite("b", sizeof (char), 1,  neirofile);
            if(i != N_N->Layers_count - 2) {     //если слой не предпоследний
                for (p = 0; p < N_N->Layers_NN[i + 1].count; ++p) {
                    fwrite(&N_N->Layers_NN[i].bias[p],
                           sizeof (NN_TYPE_D), 1,  neirofile);
                }
            }
            else {
                for (p = 0; p < N_N->Output_count; ++p) {
                    fwrite(&N_N->Layers_NN[i].bias[p],
                           sizeof (NN_TYPE_D), 1,  neirofile);
                }
            }
        }
        else {
            fwrite(" ", sizeof (char), 1,  neirofile);
        }

        //Проходим по нейронам
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {

            if(i != N_N->Layers_count - 2){
                for (t = 0; t < N_N->Layers_NN[i + 1].count; ++t) {
                    fwrite(&N_N->Layers_NN[i].N[p].weight[t],
                           sizeof (NN_TYPE_D), 1,  neirofile);
                }
            }
            else {
                for (t = 0; t < N_N->Output_count; ++t) {
                    fwrite(&N_N->Layers_NN[i].N[p].weight[t],
                           sizeof (NN_TYPE_D), 1,  neirofile);
                }
            }
        }
        --i;
    } while(i == 0);

    fclose(neirofile);
}

//Отчистка нейросети (данных, но не веса)
void NN_Clear(Neiro_Net *N_N)
{
    NN_TYPE_C i,j;
    for (i = 0; i < N_N->Layers_count - 2; ++i) {
        for (j = 0; j < N_N->Layers_NN[i].count; ++j) {
            N_N->Layers_NN[i].N->data = 0.0;
        }
    }
}

//Функция вычисления
void Result(Neiro_Net *N_N, NN_TYPE_D *input, NN_TYPE_D *output,
            NN_TYPE_D activ(NN_TYPE_D x))
{
    NN_TYPE_C i, p, t;
    //Ввод входных данных в нейроны
    for (i = 0; i < N_N->Layers_NN[0].count; ++i) {
        N_N->Layers_NN[0].N[i].data = input[i];
    }

    //Весь скрытый слой
    for(i = 0; i<N_N->Layers_count - 1; ++i){
        //Каждый нейрон этого слоя
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
            //Вычисление суммы произведений веса и данных
            for (t = 0; t < N_N->Layers_NN[i + 1].count; ++t) {
                N_N->Layers_NN[i + 1].N[t].data +=
                N_N->Layers_NN[i].N[p].weight[t] *
                N_N->Layers_NN[i].N[p].data;
            }
        }
        if(N_N->Layers_NN[i].bias != NULL) {
            for (p = 0; p < N_N->Layers_NN[i + 1].count; ++p) {
                N_N->Layers_NN[i + 1].N[p].data += N_N->Layers_NN[i].bias[p];
            }
        }
        //Активация всех нейронов слоя
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
            N_N->Layers_NN[i + 1].N[p].data =
                    activ(N_N->Layers_NN[i+1].N[p].data);
        }
    }
    //Выходной слой (просто копирование результата)
    for (p = 0; p < N_N->Output_count; ++p) {
        output[p] = N_N->Layers_NN[N_N->Layers_count - 1].N[p].data;
    }
}

//Вычисление ошибки MSE
NN_TYPE_D NN_Error_MSE(NN_TYPE_D* output_NN, NN_TYPE_D *output_Ideal,
                       const NN_TYPE_C size)
{
    NN_TYPE_C i;
    NN_TYPE_D a = 0, res = 0;
    for(i = 0; i < size; i++){
        a = output_Ideal[i] - output_NN[i];
        res += (a * a);
    }
    return res / size;
}

//Гиперпараметры
void NN_SetGlobalParam(Neiro_Net *N_N, NN_TYPE_D Alpha, NN_TYPE_D Epsilon)
{
    N_N->Alpha = Alpha;
    N_N->Epsilon = Epsilon;
}

//Обучение
void NN_Backpropagation (Neiro_Net *N_N, NN_TYPE_D *input, NN_TYPE_D *output,
                          NN_TYPE_D activ(NN_TYPE_D x))
{
    NN_TYPE_C i, p, t;
    NN_TYPE_D *NN_output = malloc(N_N->Output_count * sizeof(NN_TYPE_D));
    NN_TYPE_D Last_weight;

    NN_Clear(N_N);                          //Очищаем нейросеть
    Result(N_N, input, NN_output, activ);   //Проводим вычисления

    //Для последнего слоя
    i = N_N->Layers_count - 1;
    for (p = 0; p < N_N->Output_count; ++p) {
        N_N->Layers_NN[i].N[p].delta = output[p] - N_N->Layers_NN[i].N[p].data;
        N_N->Layers_NN[i].N[p].delta *= (output[p] - N_N->Layers_NN[i].N[p].data)
                * N_N->Layers_NN[i].N[p].data;
    }

    --i;
    //Для остальных до первого вычисляем дельту
    for (i = i; i > 0 ; --i) {
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
                N_N->Layers_NN[i].N[p].delta = 0.0;
                for (t = 0; t < N_N->Layers_NN[i + 1].count; ++t) {
                    N_N->Layers_NN[i].N[p].delta +=
                    N_N->Layers_NN[i + 1].N[t].delta *
                    N_N->Layers_NN[i].N[p].weight[t];
                }
                N_N->Layers_NN[i].N[p].delta *=
                (output[p] - N_N->Layers_NN[i].N[p].data) * N_N->Layers_NN[i].N[p].data;
        }
    }

    //Вычисление нового веса + сохранение предыдущего веса
    for (i = 0; i < N_N->Layers_count - 1; ++i) {
        for (p = 0; p < N_N->Layers_NN[i].count; ++p) {
            for (t = 0; t < N_N->Layers_NN[i + 1].count; ++t) {
                Last_weight = N_N->Layers_NN[i].N[p].weight[t];
                //градиент
                N_N->Layers_NN[i].N[p].weight[t] = N_N->Epsilon *
                (N_N->Layers_NN[i + 1].N[t].delta * N_N->Layers_NN[i].N[p].data)
                 + N_N->Alpha * N_N->Layers_NN[i].N[p].last_weight[t];

                N_N->Layers_NN[i].N[p].last_weight[t] = Last_weight;
            }
        }
    }

    NN_Clear(N_N);  //Очищаем нейросеть
    free(NN_output);
}
