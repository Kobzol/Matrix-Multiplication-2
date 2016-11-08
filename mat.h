#pragma once

#include <cstring>
#include <iostream>
#include <cassert>
#include <x86intrin.h>
#include <memory>


template <typename T>
class Matrix
{
public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        this->init();
        std::memset(this->data.get(), 0, rows * cols * sizeof(T));
    }
    Matrix(size_t rows, size_t cols, T* data) : rows(rows), cols(cols)
    {
        this->init();
        std::memcpy(this->data.get(), data, rows * cols * sizeof(T));
    }

    Matrix operator*(Matrix& rhs)
    {
        if (this->cols != rhs.rows)
        {
            throw "Invalid matrix dimensions";
        }

        Matrix result(this->cols, rhs.rows);

        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < rhs.cols; j++)
            {
                T accumulated = 0.0;
                for (int u = 0; u < this->cols; u += 4)
                {
                    accumulated += this->at(i, u) * rhs.at(u, j);
                    accumulated += this->at(i, u + 1) * rhs.at(u + 1, j);
                    accumulated += this->at(i, u + 2) * rhs.at(u + 2, j);
                    accumulated += this->at(i, u + 3) * rhs.at(u + 3, j);
                }
                result.at(i, j) = accumulated;
            }
        }

        return std::move(result);
    }
    Matrix mulTranspose(Matrix& rhs)
    {
        if (this->cols != rhs.rows)
        {
            throw "Invalid matrix dimensions";
        }

        Matrix result(this->cols, rhs.rows);
        rhs.transpose();

#pragma omp parallel for
        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < rhs.cols; j++)
            {
                T accumulated = 0.0;
                for (int u = 0; u < this->cols; u += 1)
                {
                    accumulated += this->at(i, u) * rhs.at(j, u);
                }
                result.at(i, j) = accumulated;
            }
        }

        return std::move(result);
    }
    void multiplyAccum(Matrix& rhs, Matrix& result)
    {
        if (this->cols != rhs.rows)
        {
            throw "Invalid matrix dimensions";
        }

        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < rhs.cols; j++)
            {
                T accumulated = 0.0;
                for (int u = 0; u < this->cols; u += 1)
                {
                    accumulated += this->at(i, u) * rhs.at(u, j);
                }
                result.at(i, j) += accumulated;
            }
        }
    }
    Matrix mulTransposeVec(Matrix& rhs)
    {
        if (this->cols != rhs.rows)
        {
            throw "Invalid matrix dimensions";
        }

        Matrix result(this->cols, rhs.rows);
        rhs.transpose();

        assert((long)this->data.get() % 16 == 0);
        assert((long)rhs.data.get() % 16 == 0);

        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < rhs.cols; j++)
            {
                __m128 accum = _mm_set1_ps(0.0f);
                for (int u = 0; u < this->cols; u += 4)
                {
                    __m128 m1 = _mm_load_ps(&this->at(i, u));
                    __m128 m2 = _mm_load_ps(&rhs.at(j, u));
                    accum = _mm_fmadd_ps(m1, m2, accum);
                }
                accum = _mm_hadd_ps(accum, accum);
                accum = _mm_hadd_ps(accum, accum);
                _mm_store_ss(&result.at(i, j), accum);
            }
        }

        return std::move(result);
    }

    bool operator==(Matrix& other)
    {
        if (this->rows != other.rows || this->cols != other.cols) return false;

        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < this->cols; j++)
            {
                if (std::abs(this->at(i, j) - other.at(i, j)) > 0.1f)
                {
                    return false;
                }
            }
        }

        return true;
    }

    size_t getRows() const
    {
        return this->rows;
    }
    size_t getCols() const
    {
        return this->cols;
    }

    inline T& operator[](int index)
    {
        return this->data[index];
    }
    inline T& at(int row, int col)
    {
        return this->data[row * this->cols + col];
    }
    inline const T& const_at(int row, int col) const
    {
        return this->data[row * this->cols + col];
    }

    void transpose()
    {
        for (int i = 0; i < this->rows; i++)
        {
            for (int j = i + 1; j < this->cols; j++)
            {
                std::swap(this->at(i, j), this->at(j, i));
            }
        }
    }
    void fill(bool b = false)
    {
        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < this->cols; j++)
            {
                this->at(i, j) = b ? ((i + j + 1) % 3) : (j - i % 3); // std::rand() / 1000.0;
            }
        }
    }

private:
    void init()
    {
        this->data = std::make_unique<T[]>(this->rows * this->cols);
    }

    size_t rows;
    size_t cols;
    std::unique_ptr<T[]> data;
};

template <typename T>
std::ostream& operator<<(std::ostream& o, const Matrix<T>& m)
{
    for (int i = 0; i < m.getRows(); i++)
    {
        for (int j = 0; j < m.getCols(); j++)
        {
            o << m.const_at(i, j) << " ";
        }
        o << std::endl;
    }

    return o;
}