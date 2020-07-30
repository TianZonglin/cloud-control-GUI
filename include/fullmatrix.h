#pragma once

#include <vector>



class FullMatrix
{
public:

typedef std::vector<float> Row;							//Full symmetric matrix

			 FullMatrix(int nrows_and_cols);
			~FullMatrix(); 
float&		 operator()(int i,int j);
const float& operator()(int i,int j) const;
bool		 exists(int i,int j) const;
Row&		 operator()(int i) { return *m[i]; }
const Row&	 operator()(int i) const { return *m[i]; }
 
    Row& operator[](int i) { return *m[i]; }
    const Row& operator[](int i) const { return *m[i]; }
void		 minmax();
float		 min() const { return m_min; }
float		 max() const { return m_max; }
void		 normalize();
void		 print();
int			 size() const { return m.size(); }

std::vector<Row*> m;
float		 m_min,m_max;

};



			

