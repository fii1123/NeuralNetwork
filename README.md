# NeuralNetwork
Проект прямой полносвязной искусственной нейронной сети

Структура Neiro_Net является своего рода СИшным классом.

Основными его методами являются следующие:

### Конструктор нейронки со случайными весами.

`Neiro_Net* NN_Create(NN_TYPE_C *Layers, NN_TYPE_C LayersCount, int *Biases);`

Первым параметром поступает массив, каждый элемент которого - это слой, а значение - это число нейронов.
Второй параметр - число слоев (исключая только входной. последний слой - это вывод)
Третий параметр - массив нейронов смещения. Аналогично с первым - каждый элемент - это слой, а значение - это истина или ложь.


### Инициализация нейросети из файла

`Neiro_Net* NN_Load(char *file_path);`

Единственный параметр - путь к файлу нейронки. Причем своего формата.

### Сохранение нейросети

`void NN_Save(Neiro_Net *N_N, char* file_path);`

Сохраняет в своем формате нейронку в файл.

##Отчистка нейросети (данных, но не веса)

`void NN_Clear(Neiro_Net *N_N);`


### Функция вычисления

`void Result(Neiro_Net *N_N, NN_TYPE_D *input, NN_TYPE_D *output, NN_TYPE_D activ(NN_TYPE_D x));`

### Вычисление ошибки MSE

`NN_TYPE_D NN_Error_MSE(NN_TYPE_D* output_NN, NN_TYPE_D *output_Ideal,
                       const NN_TYPE_C size);`
                       
### Настройка гиперпараметров

`void NN_SetGlobalParam(Neiro_Net *N_N, NN_TYPE_D Alpha, NN_TYPE_D Epsilon);`

### Обучение

`void NN_Backpropagation (Neiro_Net *N_N, NN_TYPE_D *input, NN_TYPE_D *output,
                          NN_TYPE_D activ(NN_TYPE_D x));`


### Удаление нейросети

`void NN_Delete(Neiro_Net *N_N);`





## Авторство и лицензия

Разработчик: Филатов И.А.
e-mail: f2000_99@mail.ru

Лицензия GNU GPL v.3
