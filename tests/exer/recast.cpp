#include <stdio.h>

void f1(int array[3][2]) {

	for (int x = 0; x < 3; ++x) {
		for (int y = 0; y < 2; ++y)
			printf("%d", array[x][y]);
	}

}
int main(int argc, char** argv) {
	//int array[2][3] =   {   { 1, 2, 3 }, { 4, 5, 6 } };
	int array[6] =   { 1, 2, 3, 4, 5, 6 };

	// Reinterpret the array with different indices
	int(*array_pointer)[3][2] = reinterpret_cast<int(*)[3][2]>(array);

	f1((*array_pointer));
/*
	for (int x = 0; x < 3; ++x) {
		for (int y = 0; y < 2; ++y)
			//printf("%d", (*array1_pointer)[x][y]);
			printf("%d", tt[x][y]);
	}
*/

	return 0;
}
