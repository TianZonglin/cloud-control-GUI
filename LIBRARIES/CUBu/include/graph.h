#pragma once


#include "include/sparsematrix.h"



class Graph : public SparseMatrix
{
public:


typedef std::pair<int,int> Edge;

			 Graph(int nnodes): SparseMatrix(nnodes) {}



};


