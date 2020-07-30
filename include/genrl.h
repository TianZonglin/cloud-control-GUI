#pragma once


const float MYINFINITY = 1.0e7f;


inline float SQR(float x)					{ return x*x; }
inline int   SQR(int x)                 	{ return x*x; }
inline float INTERP(float a,float b,float t)	{ return a*(1-t)+b*t;  }

struct Coord { 
		int i,j; 
		Coord(int i_,int j_):i(i_),j(j_) {}; 
		Coord() {} 
		int operator==(const Coord& c) const { return i==c.i && j==c.j; } 
	      };

