#pragma once

#include <vector>
#include <unordered_map>



class SparseMatrix
{
public:

typedef std::unordered_map<int,float> Row;							//Stores pairs of non-zero entries; an entry = (the col-idx,the value)

			 SparseMatrix(int nrows);
float&		 operator()(int i,int j);
const float& operator()(int i,int j) const;
bool		 exists(int i,int j) const;
Row&		 operator()(int i) { return m[i]; }
const Row&	 operator()(int i) const { return m[i]; }
void		 minmax();
float		 min() const { return m_min; }
float		 max() const { return m_max; }
void		 normalize();
int			 numRows() const { return m.size(); }

std::vector<Row> m;
float		 m_min,m_max;
};



			

