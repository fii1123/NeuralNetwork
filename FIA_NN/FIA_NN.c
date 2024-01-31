#include "FIA_NN.h"

// Универсальная функция для заполнения веса
double NN_Fill(FILE *f)
{
    double a;
    if (f == NULL) {
        return 0 ;
    }
    else {
        fread(&a, sizeof (double), 1, f);
        return a;
    }
}

// Генерация сети
Neuron_Net* NN_Generation(unsigned int *Layers, unsigned int LayersCount,
                      int *Biases, double nn_fill(FILE *f), FILE *f)
{
    unsigned int l_inx, n_inx, t;                   //Переменные
    unsigned int inx_out = LayersCount - 1;         // Индекс последнего слоя (выхода)
    Neuron_Net *N_N = malloc(sizeof (Neuron_Net));

    N_N->Alpha = 0.0;
    N_N->Epsilon = 0.0;

    //Cчитывание числа слоёв и выходных нейронов
    N_N->Layers_count = LayersCount;
    N_N->Output_count = Layers[inx_out];

    //Создание слоёв
    N_N->Layers_NN = malloc((N_N->Layers_count) * sizeof(struct Layer));

    l_inx = inx_out;

    //Обнуление последнего слоя
    N_N->Layers_NN[l_inx].N = malloc((N_N->Output_count)
                                     * sizeof(struct Neuron));
    N_N->Layers_NN[l_inx].bias = NULL;
    N_N->Layers_NN[l_inx].count = N_N->Output_count;

    for (n_inx = 0; n_inx < N_N->Output_count; ++n_inx) {
        N_N->Layers_NN[l_inx].N[n_inx].weight = NULL;
        N_N->Layers_NN[l_inx].N[n_inx].last_weight = NULL;
    }

    //Проход в обратном порядке
    --l_inx;
    do {
        N_N->Layers_NN[l_inx].count = Layers[l_inx];

        //Если есть нейрон смещения
        if(Biases != NULL && Biases[l_inx] != 0) {

            N_N->Layers_NN[l_inx].bias = malloc(sizeof(struct Neuron));
            N_N->Layers_NN[l_inx].bias->weight =
                    malloc(N_N->Layers_NN[l_inx + 1].count * sizeof(double));
            N_N->Layers_NN[l_inx].bias->last_weight =
                    malloc(N_N->Layers_NN[l_inx + 1].count * sizeof(double));

            //Заполнение весов
            for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx + 1].count; ++n_inx) {
                N_N->Layers_NN[l_inx].bias->weight[n_inx] = nn_fill(f);
                N_N->Layers_NN[l_inx].bias->last_weight[n_inx] = 0.0;
            }
        }
        else {
            N_N->Layers_NN[l_inx].bias = NULL;
        }

        N_N->Layers_NN[l_inx].N = malloc(N_N->Layers_NN[l_inx].count
                                         * sizeof(struct Neuron));
        //Проход по нейронам
        for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count; ++n_inx) {

            N_N->Layers_NN[l_inx].N[n_inx].data = 0.0;

          //Инициализация весов по числу нейронов предыдущего слоя
          //Использую переменную t для упрощения кода
            t = N_N->Layers_NN[l_inx + 1].count;

            N_N->Layers_NN[l_inx].N[n_inx].weight =
                    malloc(t * sizeof(double));
            N_N->Layers_NN[l_inx].N[n_inx].last_weight =
                    malloc(t * sizeof(double));

            //Заполнение весов
            for (t = 0; t < N_N->Layers_NN[l_inx + 1].count; ++t) {
                N_N->Layers_NN[l_inx].N[n_inx].weight[t] = nn_fill(f);
                N_N->Layers_NN[l_inx].N[n_inx].last_weight[t] = 0.0;
            }
        }
        --l_inx;
    }
    while(l_inx == 0);

    N_N->Input_count = N_N->Layers_NN[0].count;

    return N_N;
}


Neuron_Net* NN_Create(unsigned int *Layers, unsigned int LayersCount,
                      int *Biases)
{
    return NN_Generation(Layers, LayersCount, Biases, NN_Fill, NULL);
}

Neuron_Net* NN_Load(char *file_path)
{
    Neuron_Net * N_N;
    FILE *Neuronfile = fopen(file_path, "rb");
    if(Neuronfile == NULL) {
        puts("Error of open file!");
    }
    else {
        unsigned int LayersCount, l_inx;
        char bias;
        fread(&LayersCount, sizeof (unsigned int), 1,  Neuronfile);
        unsigned int *Layers = malloc(LayersCount * sizeof (unsigned int));
        int *Biases  = malloc(LayersCount * sizeof (int));

        // последний слой
        fread(&Layers[LayersCount - 1], sizeof (unsigned int), 1,  Neuronfile);
        Biases[LayersCount - 1] = 0;

        l_inx = LayersCount - 2;
        do {
            bias = 'f';
            fread(&Layers[l_inx], sizeof (unsigned int), 1,  Neuronfile);
            fread(&bias, sizeof (char), 1,  Neuronfile);
            Biases[l_inx] = (bias == 'b'); // == в Си возвращает int
            --l_inx;
        } while (l_inx == 0);

        N_N = NN_Generation(Layers, LayersCount, Biases, NN_Fill, Neuronfile);

        free(Layers);
        free(Biases);
        fclose(Neuronfile);

        return N_N;
    }
    return NULL;
}


void NN_Save(Neuron_Net *N_N, char* file_path)
{
    unsigned int l_inx, n_inx, t;                   //Переменные
    FILE *Neuronfile = fopen(file_path, "wb");

    fwrite(&N_N->Layers_count, sizeof (unsigned int), 1,  Neuronfile);
    fwrite(&N_N->Output_count, sizeof (unsigned int), 1,  Neuronfile);

    // Запись информации о слоях
    l_inx = N_N->Layers_count - 2;
    do {
        fwrite(&N_N->Layers_NN[l_inx].count,
               sizeof (unsigned int), 1,  Neuronfile);
        if (N_N->Layers_NN[l_inx].bias != NULL) {
            fwrite("b", sizeof (char), 1,  Neuronfile);
        }
        else {
            fwrite("f", sizeof (char), 1,  Neuronfile);
        }
        --l_inx;
    } while (l_inx == 0);

    // Запись данных
    l_inx = N_N->Layers_count - 2;
    do {
        // Информация о весах нейрона смещения
        if (N_N->Layers_NN[l_inx].bias != NULL) {
            /*
            */

            for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx + 1].count; ++n_inx) {
                fwrite(&N_N->Layers_NN[l_inx].bias[n_inx],
                       sizeof (double), 1,  Neuronfile);
            }
        }

        // Информация о весах прочих нейронов
        for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count; ++n_inx) {
            for (t = 0; t < N_N->Layers_NN[l_inx + 1].count; ++t) {
                fwrite(&N_N->Layers_NN[l_inx].N[n_inx].weight[t],
                       sizeof (double), 1,  Neuronfile);
            }
        }


        --l_inx;
    } while (l_inx == 0);

    fclose(Neuronfile);
}

void NN_Clear(Neuron_Net *N_N)
{
    unsigned int l_inx, n_inx;
    for (l_inx = 0; l_inx < N_N->Layers_count - 2; ++l_inx) {
        for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count; ++n_inx) {
            N_N->Layers_NN[l_inx].N[n_inx].data = 0.0;
        }
    }
}

void Result(Neuron_Net *N_N, double *input, double *output,
            double activ(double x))
{
    unsigned int l_inx, n_inx, t;
    //Ввод входных данных во входной слой
    for (n_inx = 0; n_inx < N_N->Layers_NN[0].count; ++n_inx) {
        N_N->Layers_NN[0].N[n_inx].data = input[n_inx];
    }

    //Весь скрытый слой
    for(l_inx = 0; l_inx < N_N->Layers_count - 1; ++l_inx){

        for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count; ++n_inx) {
            //Вычисление суммы произведений веса c данными
            for (t = 0; t < N_N->Layers_NN[l_inx + 1].count; ++t) {
                N_N->Layers_NN[l_inx + 1].N[t].data +=
                        N_N->Layers_NN[l_inx].N[n_inx].weight[t]
                        * N_N->Layers_NN[l_inx].N[n_inx].data;
            }

            // Данные нейрона смещения всегда равны 1
            if(N_N->Layers_NN[l_inx].bias != NULL) {
                for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx + 1].count; ++n_inx) {
                    N_N->Layers_NN[l_inx + 1].N[n_inx].data +=
                            N_N->Layers_NN[l_inx].bias->weight[n_inx];
                }
            }
        }

        //Активация всех нейронов слоя
        for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count - 1; ++n_inx) {
            N_N->Layers_NN[l_inx + 1].N[n_inx].data =
                    activ(N_N->Layers_NN[l_inx + 1].N[n_inx].data);
        }
    }



    //Вывод (просто копирование результата)
    for (n_inx = 0; n_inx < N_N->Output_count; ++n_inx) {
        output[n_inx] = N_N->Layers_NN[N_N->Layers_count - 1].N[n_inx].data;
    }
}


void NN_Backpropagation (Neuron_Net *N_N, double *input, double *output,
                          double activ(double x))
{
    unsigned int l_inx, n_inx, t;
    double *NN_output = malloc(N_N->Output_count * sizeof(double));
    double Last_weight;

    Result(N_N, input, NN_output, activ);   //Проводим вычисления

    //Для последнего слоя
    l_inx = N_N->Layers_count - 1;
    for (n_inx = 0; n_inx < N_N->Output_count; ++n_inx) {
        N_N->Layers_NN[l_inx].N[n_inx].delta =
                output[n_inx] - N_N->Layers_NN[l_inx].N[n_inx].data;
        N_N->Layers_NN[l_inx].N[n_inx].delta *=
                (output[n_inx] - N_N->Layers_NN[l_inx].N[n_inx].data) *
                N_N->Layers_NN[l_inx].N[n_inx].data;
    }

    --l_inx;
    //Для остальных до первого вычисляем дельту
    for (; l_inx > 0 ; --l_inx) {
        for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count; ++n_inx) {
                N_N->Layers_NN[l_inx].N[n_inx].delta = 0.0;
                for (t = 0; t < N_N->Layers_NN[l_inx + 1].count; ++t) {
                    N_N->Layers_NN[l_inx].N[n_inx].delta +=
                    N_N->Layers_NN[l_inx + 1].N[t].delta *
                    N_N->Layers_NN[l_inx].N[n_inx].weight[t];
                }
                N_N->Layers_NN[l_inx].N[n_inx].delta *=
                (output[n_inx] - N_N->Layers_NN[l_inx].N[n_inx].data) *
                        N_N->Layers_NN[l_inx].N[n_inx].data;
        }

        if (N_N->Layers_NN[l_inx].bias != NULL) {
            N_N->Layers_NN[l_inx].bias->delta = 0.0;
            for (t = 0; t < N_N->Layers_NN[l_inx + 1].count; ++t) {
                N_N->Layers_NN[l_inx].bias->delta +=
                        N_N->Layers_NN[l_inx + 1].N[t].delta *
                        N_N->Layers_NN[l_inx].bias->weight[t];
            }
            N_N->Layers_NN[l_inx].bias->delta *= output[n_inx] - 1;
        }


    }

    //Вычисление нового веса + сохранение предыдущего веса
    for (l_inx = 0; l_inx < N_N->Layers_count - 1; ++l_inx) {
        for (t = 0; t < N_N->Layers_NN[l_inx + 1].count; ++t) {
            for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count; ++n_inx) {

                Last_weight = N_N->Layers_NN[l_inx].N[n_inx].weight[t];
                //градиент
                N_N->Layers_NN[l_inx].N[n_inx].weight[t] = N_N->Epsilon
                        * (N_N->Layers_NN[l_inx + 1].N[t].delta
                        * N_N->Layers_NN[l_inx].N[n_inx].data)
                 + N_N->Alpha * N_N->Layers_NN[l_inx].N[n_inx].last_weight[t];

                N_N->Layers_NN[l_inx].N[n_inx].last_weight[t] = Last_weight;
            }
            if (N_N->Layers_NN[l_inx].bias != NULL) {
                Last_weight = N_N->Layers_NN[l_inx].bias->weight[t];
                N_N->Layers_NN[l_inx].bias->weight[t] =
                        N_N->Epsilon * N_N->Layers_NN[l_inx + 1].N[t].delta
                        + N_N->Alpha * N_N->Layers_NN[l_inx].bias->last_weight[t];
                N_N->Layers_NN[l_inx].bias->last_weight[t] = Last_weight;
            }
        }
    }

    NN_Clear(N_N);  //Очищаем нейросеть
    free(NN_output);
}


void NN_Delete(Neuron_Net *N_N)
{
    unsigned int l_inx, n_inx;
    for (int l_inx = 0; l_inx < N_N->Layers_count - 1; ++l_inx) {
        for (n_inx = 0; n_inx < N_N->Layers_NN[l_inx].count; ++n_inx) {
            free(N_N->Layers_NN[l_inx].N[n_inx].weight);
            free(N_N->Layers_NN[l_inx].N[n_inx].last_weight);
        }
        if (N_N->Layers_NN[l_inx].bias != NULL) {
            free(N_N->Layers_NN[l_inx].bias->weight);
            free(N_N->Layers_NN[l_inx].bias->last_weight);
            free(N_N->Layers_NN[l_inx].bias);
        }
        free(N_N->Layers_NN[l_inx].N);
    }
    free(N_N->Layers_NN);
    free(N_N);
}
