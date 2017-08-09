/*
 * cudamatrix_types.cuh
 *
 *  Created on: Aug 4, 2010
 *      Author: paynej
 */

#include "cudamatrix.cuh"

typedef class matrix4T<double> matrix4d;
typedef class cudaMatrixT<double> cudaMatrix;
typedef class cudaMatrixT<double> cudaMatrixd;
typedef class cudaMatrixT<double2> cudaMatrixd2;
typedef class cudaMatrixT<float> cudaMatrixf;
typedef class cudaMatrixT<int> cudaMatrixi;
typedef class cudaMatrixT<int2> cudaMatrixi2;
typedef class cudaMatrixT<int3> cudaMatrixi3;
typedef class cudaMatrixT<int4> cudaMatrixi4;
typedef class cudaMatrixT<matrix4d> cudaMatrixM4;
