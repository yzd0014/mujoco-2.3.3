#pragma once
#include <mujoco/mujoco.h>
#include <Eigen/Eigen>

using namespace Eigen;

void CopyVectorMjtoEigen(mjtNum* i_v, VectorXd& o_v, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		o_v(i) = i_v[i];
	}
}

void CopyVectorEigentoMj(VectorXd& i_v, mjtNum* o_v, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		o_v[i] = i_v(i);
	}
}

void CopyMatrixMjtoEigen(mjtNum* i_m, MatrixXd& o_m, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			o_m(i, j) = i_m[i * col + j];
		}
	}
}

void CopyMatrixEigentoMj(MatrixXd& i_m, mjtNum* o_m, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			o_m[i * col + j] = i_m(i, j);
		}
	}
}