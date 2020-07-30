#include "include/fullmatrix.h"
#include <utility>
#include <stdio.h>
using namespace std;





FullMatrix::FullMatrix(int nrows_and_cols):m_min(0),m_max(0)
{
	m.resize(nrows_and_cols);
	for(int i=0;i<nrows_and_cols;++i)
	   m[i] = new Row(nrows_and_cols);
}

FullMatrix::~FullMatrix()
{
	for(int i=0;i<m.size();++i)
	   delete m[i];
}

void FullMatrix::normalize()
{
	float range = m_max-m_min;
	for(int i=0,N=m.size();i<N;++i)
	{
		Row& row = *m[i];
		for(Row::iterator it=row.begin();it!=row.end();++it)
		{
			*it = (*it-m_min)/range;
		}
	}	
	
	m_min=0; m_max = 1;
}


void FullMatrix::print(){
	for(int i=0,N=m.size();i<N;++i)
	{
		const Row& row = *m[i];
		for(Row::const_iterator it=row.begin();it!=row.end();++it)
		{
			float val = *it;
			printf("%f ",val);
		}
		printf("\n");
	}

}

void FullMatrix::minmax()
{
	for(int i=0,N=m.size();i<N;++i)
	{
		const Row& row = *m[i];
		for(Row::const_iterator it=row.begin();it!=row.end();++it)
		{
			float val = *it;
			m_min = std::min(m_min,val);
			m_max = std::max(m_max,val);
		}
	}	
}

float& FullMatrix::operator()(int i,int j)
{
	Row& row = *m[i];								//get the row (should always exist..)
	return row[j];
}


bool FullMatrix::exists(int i,int j) const
{
	return true;
}


const float& FullMatrix::operator()(int i,int j) const
{
	const Row& row = *m[i];							//get the row (should always exist..)
	return row[j];
}
	

 