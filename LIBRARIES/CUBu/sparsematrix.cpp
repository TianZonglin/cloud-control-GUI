#include "include/sparsematrix.h"
#include <utility>

using namespace std;



static const float dummy = 0;



SparseMatrix::SparseMatrix(int nrows):m_min(0),m_max(0)
{
	m.resize(nrows);
}

void SparseMatrix::normalize()
{
	float range = m_max-m_min;
	for(int i=0,N=m.size();i<N;++i)
	{
		Row& row = m[i];
		for(Row::iterator it=row.begin();it!=row.end();++it)
		{
			it->second = (it->second-m_min)/range;
		}
	}	
	
	m_min=0; m_max = 1;
}


void SparseMatrix::minmax()
{
	for(int i=0,N=m.size();i<N;++i)
	{
		const Row& row = m[i];
		for(Row::const_iterator it=row.begin();it!=row.end();++it)
		{
			float val = it->second;
			m_min = std::min(m_min,val);
			m_max = std::max(m_max,val);
		}
	}	
}

float& SparseMatrix::operator()(int i,int j)
{
	Row& row = m[i];								//get the row (should always exist..)
	
	Row::iterator it = row.find(j);
	
	if (it==row.end())
	{
		it = row.insert(make_pair(j,0)).first;
		return (*it).second;
	}
	else
	{
		return (*it).second;
	}
}


bool SparseMatrix::exists(int i,int j) const
{
	const Row& row = m[i];							//get the row (should always exist..)
	
	Row::const_iterator it = row.find(j);
	
	return it!=row.end();
}


const float& SparseMatrix::operator()(int i,int j) const
{
	const Row& row = m[i];							//get the row (should always exist..)
	
	Row::const_iterator it = row.find(j);
	
	if (it==row.end())
	{
		return dummy;
	}
	else
	{
		return (*it).second;
	}
}
	


