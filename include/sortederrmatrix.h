#pragma once
#include <map>
#include <vector>




class SortedErrorMatrix
{
public:

typedef std::multimap<float,int> Row;

								 SortedErrorMatrix(int n) { m.resize(n); }
const Row&						 operator()(int i) const  { return m[i]; }				
Row&							 operator()(int i)        { return m[i]; }				

std::vector<Row>				 m;
};






	

