#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <fstream>

using namespace std;

int **allocateMatrix(int n)
{
    int *data = (int *)malloc(n * n * sizeof(int));
    int **array = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]);
    }
    return array;
}

void fillMatrix(int n, int **&mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i][j] = rand() % 5;
        }
    }
}

void freeMatrix(int n, int **mat)
{
    free(mat[0]);
    free(mat);
}

int **naive(int n, int **mat1, int **mat2)
{
    int **prod = allocateMatrix(n);

    int i, j;

#pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            prod[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                prod[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return prod;
}

int **getSlice(int n, int **mat, int offseti, int offsetj)
{
    int m = n / 2;
    int **slice = allocateMatrix(m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = mat[offseti + i][offsetj + j];
        }
    }
    return slice;
}

int **addMatrices(int n, int **mat1, int **mat2, bool add)
{
    int **result = allocateMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (add)
                result[i][j] = mat1[i][j] + mat2[i][j];
            else
                result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    return result;
}

int **combineMatrices(int m, int **c11, int **c12, int **c21, int **c22)
{
    int n = 2 * m;
    int **result = allocateMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j];
            else if (i < m)
                result[i][j] = c12[i][j - m];
            else if (j < m)
                result[i][j] = c21[i - m][j];
            else
                result[i][j] = c22[i - m][j - m];
        }
    }

    return result;
}

int **strassen(int n, int **mat1, int **mat2)
{

    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    int m = n / 2;

    int **a = getSlice(n, mat1, 0, 0);
    int **b = getSlice(n, mat1, 0, m);
    int **c = getSlice(n, mat1, m, 0);
    int **d = getSlice(n, mat1, m, m);
    int **e = getSlice(n, mat2, 0, 0);
    int **f = getSlice(n, mat2, 0, m);
    int **g = getSlice(n, mat2, m, 0);
    int **h = getSlice(n, mat2, m, m);

    int **m1;
#pragma omp task shared(m1)
    {
        int **bds = addMatrices(m, b, d, false);
        int **gha = addMatrices(m, g, h, true);
        m1 = strassen(m, bds, gha);
        freeMatrix(m, bds);
        freeMatrix(m, gha);
    }

    int **m2;
#pragma omp task shared(m2)
    {
        int **ada = addMatrices(m, a, d, true);
        int **eha = addMatrices(m, e, h, true);
        m2 = strassen(m, ada, eha);
        freeMatrix(m, ada);
        freeMatrix(m, eha);
    }

    int **m3;
#pragma omp task shared(m3)
    {
        int **acs = addMatrices(m, a, c, false);
        int **efa = addMatrices(m, e, f, true);
        m3 = strassen(m, acs, efa);
        freeMatrix(m, acs);
        freeMatrix(m, efa);
    }

    int **m4;
#pragma omp task shared(m4)
    {
        int **aba = addMatrices(m, a, b, true);
        m4 = strassen(m, aba, h);
        freeMatrix(m, aba);
    }

    int **m5;
#pragma omp task shared(m5)
    {
        int **fhs = addMatrices(m, f, h, false);
        m5 = strassen(m, a, fhs);
        freeMatrix(m, fhs);
    }

    int **m6;
#pragma omp task shared(m6)
    {
        int **ges = addMatrices(m, g, e, false);
        m6 = strassen(m, d, ges);
        freeMatrix(m, ges);
    }

    int **m7;
#pragma omp task shared(m7)
    {
        int **cda = addMatrices(m, c, d, true);
        m7 = strassen(m, cda, e);
        freeMatrix(m, cda);
    }

#pragma omp taskwait

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    int **c11;
#pragma omp task shared(c11)
    {
        int **s1s2a = addMatrices(m, m1, m2, true);
        int **s6s4s = addMatrices(m, m6, m4, false);
        c11 = addMatrices(m, s1s2a, s6s4s, true);
        freeMatrix(m, s1s2a);
        freeMatrix(m, s6s4s);
    }

    int **c12;
#pragma omp task shared(c12)
    {
        c12 = addMatrices(m, m4, m5, true);
    }

    int **c21;
#pragma omp task shared(c21)
    {
        c21 = addMatrices(m, m6, m7, true);
    }

    int **c22;
#pragma omp task shared(c22)
    {
        int **s2s3s = addMatrices(m, m2, m3, false);
        int **s5s7s = addMatrices(m, m5, m7, false);
        c22 = addMatrices(m, s2s3s, s5s7s, true);
        freeMatrix(m, s2s3s);
        freeMatrix(m, s5s7s);
    }

#pragma omp taskwait

    freeMatrix(m, m1);
    freeMatrix(m, m2);
    freeMatrix(m, m3);
    freeMatrix(m, m4);
    freeMatrix(m, m5);
    freeMatrix(m, m6);
    freeMatrix(m, m7);

    int **prod = combineMatrices(m, c11, c12, c21, c22);

    freeMatrix(m, c11);
    freeMatrix(m, c12);
    freeMatrix(m, c21);
    freeMatrix(m, c22);

    return prod;
}

bool check(int n, int **prod1, int **prod2)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (prod1[i][j] != prod2[i][j])
                return false;
        }
    }
    return true;
}

void printMatrix(int n, int ** mat, ostream& stream)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            stream << mat[i][j] << " ";
        }
        stream << endl;
    }
    stream << endl;
}

void outputToFile(int n, int **a, int **b, int **result)
{
    ofstream out;
    out.open("result.txt");
    if (out.is_open())
    {
        out << "Matrix A:" << endl;
        printMatrix(n, a, out);
        out << "Matrix B:" << endl;
        printMatrix(n, a, out);
        out << "Multiplication Result:" << endl;
        printMatrix(n, a, out);
    }
    out.close(); 
    std::cout << "File has been written" << std::endl;
}

int main()
{
    int n;
    cout << "\nEnter matrix Dimension: ";
    cin >> n;

    int **mat1 = allocateMatrix(n);
    fillMatrix(n, mat1);

    int **mat2 = allocateMatrix(n);
    fillMatrix(n, mat2);

    int threadCounts[] = {1, 2, 4, 8, 12, 16, 32, 64, 128, 256};
    int **prod;

    for (int threads : threadCounts)
    {
        double startParStrassen = omp_get_wtime();

        omp_set_num_threads(threads);

        #pragma omp parallel
        {
        #pragma omp single
            {
                prod = strassen(n, mat1, mat2);
            }
        }
        double endParStrassen = omp_get_wtime();
        cout << "\nParallel Strassen (" << threads << " threads) Runtime: ";
        cout << setprecision(5) << endParStrassen - startParStrassen << endl;
        cout << endl;
    }
    outputToFile(n, mat1, mat2, prod);
    system("pause");
    return 0;
}