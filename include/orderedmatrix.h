#pragma once

#include <vector>



template <class T> class OrderedMatrix
{
public:

typedef std::vector<T> Row;

			 OrderedMatrix(int nrows)				{ m.resize(nrows); }
T&			 operator()(int i,int j)				{ return m[i][j]; }
const T&	 operator()(int i,int j) const			{ return m[i][j]; }
Row&		 operator()(int i)						{ return m[i]; }
const Row&	 operator()(int i) const				{ return m[i]; }

int			 next(int i,int j) const { return m[i][(j+1) % m[i].size()]; }				//Find next edge end-point, clockwise, from edge 'j' of point 'i'
int			 prev(int i,int j) const { return m[i][(j-1+m[i].size()) % m[i].size()];}	//Find next edge end-point, clockwise, from edge 'j' of point 'i'
int			 idx(int i,int val) const
			 {
				const Row& r = m[i];
				for(int j=0,N=r.size();j<N;++j)
				   if (r[j]==val)	return j;
				return -1;   
			 }

std::vector<Row> m;
};



			

