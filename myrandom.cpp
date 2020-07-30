#include "include/myrandom.h"
#include <limits.h>


//**********************************************************************************************
//* Random Point Generator
//**********************************************************************************************


static unsigned long z, w, jsr, jcong; // Seeds


inline unsigned long znew() 
{ return (z = 36969 * (z & 0xfffful) + (z >> 16)); }

inline unsigned long wnew() 
{ return (w = 18000 * (w & 0xfffful) + (w >> 16)); }

inline unsigned long MWC()  
{ return ((znew() << 16) + wnew()); }

inline unsigned long SHR3()
{ jsr ^= (jsr << 17); jsr ^= (jsr >> 13); return (jsr ^= (jsr << 5)); }

inline unsigned long CONG() 
{ return (jcong = 69069 * jcong + 1234567); }

inline unsigned long rand_int()         // [0,2^32-1]
{ return ((MWC() ^ CONG()) + SHR3()); }



double myrandom()     // [0,1)
{ return ((double) rand_int() / (double(ULONG_MAX)+1)); }

void randinit(unsigned long x_) 
{ z =x_; w = x_; jsr = x_; jcong = x_; }

