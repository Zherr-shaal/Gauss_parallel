// ПСРСЛаба3.1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include "omp.h"
#include <ctime>

struct SLAU
{
	double ** koef; // матрица коэффициентов
	double * X; // матрица неизвестных
	double * Y; // матрица значений
	int m; // число неизвестных в СЛАУ
	int n; // число уравнений в СЛАУ
};

void print_SLAU(SLAU sys) {//Печать матрицы
	std::cout << "Вывод матрицы:\n  ";
	for (int i = 0; i < sys.m; i++) {
		std::cout << i + 1 << " ";
	}
	std::cout << "Y\n";
	for (int i = 0; i < sys.n; i++) {
		std::cout << i + 1 << " ";
		for (int j = 0; j < sys.m; j++) {
			std::cout << sys.koef[i][j] << " ";
		}
		std::cout << sys.Y[i] << "\n";
	}
}
void swap(double * el1, double * el2) {
	double s = *el1;
	*el1 = *el2;
	*el2 = s;
}
void swap_lines(double *l1, double *l2,int m) {//Расширение swap для строк матрицы
	for (int i = 0; i < m; i++) {
		swap(&l1[i], &l2[i]);
	}
}
void SLAU_swap_lines(SLAU sys, int l1, int l2) {//Необходимо выполнить перестановку и для матрицы свободных членов
	swap_lines(sys.koef[l1], sys.koef[l2],sys.m);
	swap(&sys.Y[l1], &sys.Y[l2]);
}
int SLAU_max(SLAU sys, int row, int col) {//Поиск максимума по столбцу
	int max_num = row;
	double max = sys.koef[row][col];
	for (int i = row; i < sys.n; i++) {
		if (sys.koef[i][col] > max) {
			max_num = i;
			max = sys.koef[i][col];
		}
	}
	return max_num;
}

void SLAU_norm(SLAU sys, int row, int col) {//Нормировка

	for (int i = row; i < sys.n; i++) {
		double norm = sys.koef[i][col];
		for (int j = col; j < sys.m; j++) {
			sys.koef[i][j] = sys.koef[i][j] / norm;
		}
		sys.Y[i] = sys.Y[i] / norm;
	}
}
void SLAU_norm_parallel(SLAU sys, int row, int col) {//Паралелльная нормировка
#pragma omp parallel for
	for (int i = row; i < sys.n; i++) {
		double norm = sys.koef[i][col];
		for (int j = col; j < sys.m; j++) {
			sys.koef[i][j] = sys.koef[i][j] / norm;
		}
		sys.Y[i] = sys.Y[i] / norm;
	}
}
double* substract_lines(double *l1, double *l2, int m) {//Вычитание строки из строки
	double *result = new double[m];
	for (int i = 0; i < m; i++) {
		result[i] = l1[i] - l2[i];
	}
	return result;
}


void SLAU_substraction_line(SLAU sys, int row) {//основной цикл вычитания
	double* sub_koef = sys.koef[row];
	double sub_y = sys.Y[row];
	for (int i = row+1; i < sys.n; i++) {
		sys.koef[i]=substract_lines(sys.koef[i], sub_koef, sys.m);
		sys.Y[i]-= sub_y;
	}
}

void SLAU_substraction_line_parallel(SLAU sys, int row) {//паралелльный цикл вычитания
	double* sub_koef = sys.koef[row];
	double sub_y = sys.Y[row];
#pragma omp parallel for
	for (int i = row + 1; i < sys.n; i++) {
		for (int j = 0; j < sys.m; j++) {
			 sys.koef[i][j] -= sub_koef[j];
		}
		sys.Y[i] -= sub_y;
	}
}
void SLAU_transform(SLAU sys,bool par) {//Прямой ход метода Гаусса
	for (int i = 0, j = 0; i < sys.n-1&&j<sys.m-1; i++, j++) {
		SLAU_swap_lines(sys, i, SLAU_max(sys, i, j));

		if (par){
			SLAU_norm_parallel(sys, i, j);
			SLAU_substraction_line_parallel(sys, i);
		}
		else {
			SLAU_norm(sys, i, j);
			SLAU_substraction_line(sys, i);
		}
	}
}
void SLAU_get_solution(SLAU sys) {//Обратный ход метода Гаусса
	sys.X[sys.m-1] = sys.Y[sys.n-1] / sys.koef[sys.n-1][sys.m-1];
	for (int i = sys.m - 2,k=sys.n-2; i >= 0&&k>=0; i--,k--) {
		double x_sum = 0;
		for (int j = sys.m - 1; j > i; j--) {
			x_sum += sys.X[j]* sys.koef[k][j];
		}
		sys.X[i] = (sys.Y[i] - x_sum) / sys.koef[k][i];
	}
}
void print_solution(SLAU sys) {//Печать решений
	std::cout << "Решение:\n";
	for (int i = 0; i < sys.m; i++) {
		std::cout << sys.X[i] << " ";
	}
}
SLAU get_SLAU_rand(int n, int m) {//получение случайной СЛАУ
	srand(time(NULL));
	SLAU res;
	res.X = new double[m];
	res.Y = new double[n];
	res.n = n;
	res.m = m;
	res.koef = new double*[n];
	for (int i = 0; i < n; i++) {
		res.koef[i] = new double[m];
		for (int j = 0; j < m; j++) {
			res.koef[i][j] = (double)rand() / RAND_MAX * (100 + 100);
		}
		res.Y[i] = (double)rand() / RAND_MAX * (100 + 100);
	}
	return res;
}
int Kroneker_Kapelli(SLAU sys) {//Проверка теоремы Кронекера-Капелли, если вдруг попалась матрица с линейно зависимыми строками
	if (sys.koef[sys.n-1][sys.m-1] != 0) {
		return 0;
	}
	else {
		if (sys.Y[sys.n-1] != 0) {
			return 1;
		}
		else {
			return 2;
		}
	}
}
SLAU get_SLAU_hand(int n, int m) {//Ручной ввод СЛАУ
	SLAU res;
	res.X = new double[m];
	res.Y = new double[n];
	res.n = n;
	res.m = m;
	res.koef = new double*[n];
	std::cout << "Ввод матрицы:\n  ";
	for (int i = 0; i < m; i++) {
		std::cout << i+1 << " ";
	}
	std::cout << "Y\n";
	for (int i = 0; i < n; i++) {
		res.koef[i] = new double[m];
		std::cout <<i+1<<" ";
		for (int j = 0; j < m; j++) {
			std::cin >> res.koef[i][j];
		}
		std::cin >> res.Y[i];
	}
	return res;
}
int main()
{
	system("chcp 1251 > Nul");
	int n, m;
	std::cout << "Введите количество уравнений: ";
	std::cin >> n;
	std::cout << "Введите количество неизвестных: ";
	std::cin >> m;
	SLAU sys1 = get_SLAU_rand(n, m);
	SLAU sys2 = get_SLAU_rand(n, m);
	/*print_SLAU(sys1);*/
	clock_t t = clock();
	SLAU_transform(sys1, false);
	switch (Kroneker_Kapelli(sys1)) {
	case 0:
		SLAU_get_solution(sys1);
		t = clock() - t;
		/*print_solution(sys1);*/
		break;
	case 1:
		std::cout << "Система несовместна";
		t = clock() - t;
		break;
	case 2:
		std::cout << "Система имеет бесконечное множество решений";
		t = clock() - t;
		break;
	}

	std::cout << "\nЗатрачено времени (последовательно): " << (((double)t) / CLOCKS_PER_SEC);
	/*print_SLAU(sys2);*/
	t = clock();
	SLAU_transform(sys2, true);
	switch (Kroneker_Kapelli(sys2)) {
	case 0:
		SLAU_get_solution(sys2);
		t = clock() - t;
		/*print_solution(sys2);*/
		break;
	case 1:
		std::cout << "Система несовместна";
		t = clock() - t;
		break;
	case 2:
		std::cout << "Система имеет бесконечное множество решений";
		t = clock() - t;
		break;
	}
	std::cout << "\nЗатрачено времени (параллельно):     " << (((double)t) / CLOCKS_PER_SEC);
	for (int i = 0; i < sys1.n; i++) {
		delete[] sys1.koef[i];
	}
	for (int i = 0; i < sys2.n; i++) {
		delete[] sys2.koef[i];
	}
	delete[] sys1.koef;
	delete[] sys2.koef;
	delete[] sys1.X;
	delete[] sys2.X;
	delete[] sys1.Y;
	delete[] sys2.Y;
	return 0;
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
