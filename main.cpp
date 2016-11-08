#include <mpi.h>
#include <cmath>
#include "mat.h"

#define TAG_MATRIX_A_DISTRIBUTE (1)
#define TAG_MATRIX_B_DISTRIBUTE (2)
#define TAG_MATRIX_A_SHIFT (3)
#define TAG_MATRIX_B_SHIFT (4)


void shiftLeft(Matrix<double>& mat, int subMatrixWidth)
{
    std::unique_ptr<double[]> buffer = std::make_unique<double[]>(mat.getCols());

    unsigned int gap = 0;
    for (int i = subMatrixWidth; i < mat.getRows(); i++)
    {
        if (gap % subMatrixWidth == 0)
        {
            gap++;
        }

        for (int j = 0; j < mat.getCols(); j++)
        {
            size_t previous = ((j - (gap * subMatrixWidth)) + mat.getCols()) % mat.getCols();
            buffer[previous] = mat.at(i, j);
        }

        memcpy(&mat.at(i, 0), buffer.get(), sizeof(double) * mat.getCols());
    }
}
void shiftUp(Matrix<double>& mat, int subMatrixWidth)
{
    std::unique_ptr<double[]> buffer = std::make_unique<double[]>(mat.getCols());

    unsigned int gap = 0;
    for (int j = subMatrixWidth; j < mat.getCols(); j++)
    {
        if (j % subMatrixWidth == 0)
        {
            gap++;
        }

        for (int i = 0; i < mat.getRows(); i++)
        {
            size_t next = ((i - (gap * subMatrixWidth)) + mat.getRows()) % mat.getRows();
            buffer[next] = mat.at(i, j);
        }

        for (int i = 0; i < mat.getRows(); i++)
        {
            mat.at(i, j) = buffer[i];
        }
    }
}

void loadSubmatrix(Matrix<double>& mat, double* result, int x, int y, int subMatrixWidth)
{
    for (int i = 0; i < subMatrixWidth; i++)
    {
        for (int j = 0; j < subMatrixWidth; j++)
        {
            *result++ = mat.at(x + i, y + j);
        }
    }
}
void storeSubmatrix(Matrix<double>& result, double* input, int x, int y, int subMatrixWidth)
{
    for (int i = 0; i < subMatrixWidth; i++)
    {
        for (int j = 0; j < subMatrixWidth; j++)
        {
            result.at(x + i, y + j) = *input++;
        }
    }
}
void initialDistribute(Matrix<double>& a, Matrix<double>& b, size_t workerCount)
{
    int workerWidth = (int) std::sqrt(workerCount);
    int subMatrixWidth = (int) a.getRows() / workerWidth;
    int subMatrixSize = subMatrixWidth * subMatrixWidth;

    std::unique_ptr<double[]> buffer = std::make_unique<double[]>((size_t) subMatrixSize);
    for (int i = 0; i < workerWidth; i++)
    {
        for (int j = 0; j < workerWidth; j++)
        {
            loadSubmatrix(a, buffer.get(), i * subMatrixWidth, j * subMatrixWidth, subMatrixWidth);
            MPI_Send(buffer.get(), subMatrixSize, MPI_DOUBLE, i * workerWidth + j + 1, TAG_MATRIX_A_DISTRIBUTE, MPI_COMM_WORLD);
            loadSubmatrix(b, buffer.get(), i * subMatrixWidth, j * subMatrixWidth, subMatrixWidth);
            MPI_Send(buffer.get(), subMatrixSize, MPI_DOUBLE, i * workerWidth + j + 1, TAG_MATRIX_B_DISTRIBUTE, MPI_COMM_WORLD);
        }
    }

}
void receiveResult(Matrix<double>& result, size_t workerCount)
{
    int workerWidth = (int) std::sqrt(workerCount);
    int subMatrixWidth = (int) result.getRows() / workerWidth;

    std::unique_ptr<double[]> buffer = std::make_unique<double[]>((size_t) subMatrixWidth * subMatrixWidth);
    MPI_Status status;
    for (int i = 0; i < workerWidth; i++)
    {
        for (int j = 0; j < workerWidth; j++)
        {
            int process = i * workerWidth + j + 1;
            MPI_Recv(buffer.get(), subMatrixWidth * subMatrixWidth, MPI_DOUBLE, process, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            storeSubmatrix(result, buffer.get(), i * subMatrixWidth, j * subMatrixWidth, subMatrixWidth);
        }
    }
}

void master(size_t matrixWidth, size_t workerCount)
{
    Matrix<double> matrixA(matrixWidth, matrixWidth);
    Matrix<double> matrixB(matrixWidth, matrixWidth);
    Matrix<double> matrixTest(matrixWidth, matrixWidth);
    Matrix<double> matrixResult(matrixWidth, matrixWidth);

    matrixA.fill();
    matrixB.fill(true);
    matrixA.multiplyAccum(matrixB, matrixTest);

    int workerWidth = (int) std::sqrt(workerCount);
    int subMatrixWidth = (int) matrixWidth / workerWidth;

    shiftLeft(matrixA, subMatrixWidth);
    shiftUp(matrixB, subMatrixWidth);

    initialDistribute(matrixA, matrixB, workerCount);
    receiveResult(matrixResult, workerCount);

    std::cout << matrixTest << std::endl;
    std::cout << matrixResult << std::endl;

    if (!(matrixTest == matrixResult))
    {
        std::cout << "Not equal" << std::endl;
    }
    else std::cout << "Equal" << std::endl;
}

int pos_to_rank(int* pos, int workerWidth)
{
    return (pos[0] * workerWidth + pos[1]) + 1; // + 1 because of master
}
void worker(size_t matrixWidth, int rank, size_t workerCount)
{
    int workerWidth = (int) std::sqrt(workerCount);
    size_t subMatrixWidth = (size_t) matrixWidth / workerWidth;
    int subMatrixSize = (int) subMatrixWidth * (int) subMatrixWidth;

    Matrix<double> matrixA(subMatrixWidth, subMatrixWidth);
    Matrix<double> matrixB(subMatrixWidth, subMatrixWidth);
    Matrix<double> matrixResult(subMatrixWidth, subMatrixWidth);

    // initial matrix distribution
    MPI_Status status;
    MPI_Recv(&matrixA[0], subMatrixSize, MPI_DOUBLE, 0, TAG_MATRIX_A_DISTRIBUTE, MPI_COMM_WORLD, &status);
    MPI_Recv(&matrixB[0], subMatrixSize, MPI_DOUBLE, 0, TAG_MATRIX_B_DISTRIBUTE, MPI_COMM_WORLD, &status);

    rank--; // because of master we decrement the rank
    int posX = rank / workerWidth;
    int posY = rank % workerWidth;
    // left, up, right, down
    int neighbours[4][2] = {
            { posX, ((posY - 1) + workerWidth) % workerWidth },
            { ((posX - 1) + workerWidth) % workerWidth, posY },
            { posX, ((posY + 1) + workerWidth) % workerWidth },
            { ((posX + 1) + workerWidth) % workerWidth, posY }
    };

    for (int i = 0; i < workerWidth; i++)
    {
        matrixA.multiplyAccum(matrixB, matrixResult);

        int left = pos_to_rank(neighbours[0], workerWidth);
        int up = pos_to_rank(neighbours[1], workerWidth);
        int right = pos_to_rank(neighbours[2], workerWidth);
        int down = pos_to_rank(neighbours[3], workerWidth);

//        MPI_Send(&matrixA[0], subMatrixSize, MPI_DOUBLE, left, TAG_MATRIX_A_SHIFT, MPI_COMM_WORLD);
//        MPI_Send(&matrixB[0], subMatrixSize, MPI_DOUBLE, up, TAG_MATRIX_B_SHIFT, MPI_COMM_WORLD);
//
//        MPI_Recv(&matrixA[0], subMatrixSize, MPI_DOUBLE, right, TAG_MATRIX_A_SHIFT, MPI_COMM_WORLD, &status);
//        MPI_Recv(&matrixB[0], subMatrixSize, MPI_DOUBLE, down, TAG_MATRIX_B_SHIFT, MPI_COMM_WORLD, &status);

        MPI_Sendrecv_replace(&matrixA[0], subMatrixSize, MPI_DOUBLE,
                             left, TAG_MATRIX_A_SHIFT,
                             right, TAG_MATRIX_A_SHIFT,
                             MPI_COMM_WORLD, &status);
        MPI_Sendrecv_replace(&matrixB[0], subMatrixSize, MPI_DOUBLE,
                             up, TAG_MATRIX_B_SHIFT,
                             down, TAG_MATRIX_B_SHIFT,
                             MPI_COMM_WORLD, &status);
    }

    // send to master
    MPI_Send(&matrixResult[0], subMatrixSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{
    std::srand((unsigned int) time(NULL));

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int matrixWidth = 16;

    // matrixWidth must be a perfect square
    assert((int) std::sqrt(matrixWidth) * (int) std::sqrt(matrixWidth) == matrixWidth);
    // node count must be a perfect square
    assert((int) std::sqrt(size - 1) * (int) std::sqrt(size - 1) == (size - 1));

    if (rank == 0)
    {
        master(matrixWidth, (size_t) size - 1);
    }
    else worker(matrixWidth, rank, (size_t) size - 1);

    MPI_Finalize();

    return 0;
}