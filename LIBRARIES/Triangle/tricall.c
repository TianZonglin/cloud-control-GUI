
#include <stdio.h>


#define REAL double
#include "triangle.h"

#ifndef _STDLIB_H_
extern void *malloc();
extern void free();
#endif /* _STDLIB_H_ */



/*****************************************************************************/
/*                                                                           */
/*  main()   Create and refine a mesh.                                       */
/*                                                                           */
/*****************************************************************************/

int main()
{
  struct triangulateio in, mid; int i;

  /* Define input points. */

  in.numberofpoints = 6;
  in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
  in.pointlist[0] = 0.0;
  in.pointlist[1] = 0.0;
  in.pointlist[2] = 1.0;
  in.pointlist[3] = 0.0;
  in.pointlist[4] = 1.0;
  in.pointlist[5] = 10.0;
  in.pointlist[6] = 0.0;
  in.pointlist[7] = 10.0;
  in.pointlist[8] = 1.0;
  in.pointlist[9] = 5.0;
  in.pointlist[10]= 0.5;
  in.pointlist[11]= 5.0;
  
  in.numberofpointattributes = in.numberoftriangleattributes = 0;
  in.numberoftriangles = 0;
  in.pointmarkerlist = (int*) malloc(in.numberofpoints *sizeof(int));
  for(i=0;i<in.numberofpoints;i++) in.pointmarkerlist[i] = 100+i;

  in.numberofsegments = 7;
  in.segmentlist = (int*) malloc(in.numberofsegments * 2 * sizeof(int));
  in.segmentlist[0] = 0;
  in.segmentlist[1] = 1;
  in.segmentlist[2] = 1;
  in.segmentlist[3] = 4;
  in.segmentlist[4] = 4;
  in.segmentlist[5] = 5;
  in.segmentlist[6] = 5;
  in.segmentlist[7] = 4;
  in.segmentlist[8] = 4;
  in.segmentlist[9] = 2;
  in.segmentlist[10] = 2;
  in.segmentlist[11] = 3;
  in.segmentlist[12] = 3;
  in.segmentlist[13] = 0;

  in.segmentmarkerlist = NULL; in.holelist = NULL; in.regionlist = NULL;
  in.numberofholes = in.numberofregions = 0;
  
  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `mid'  */

  mid.pointlist = (REAL *) NULL;            
  mid.pointmarkerlist = (int *) NULL; 
  mid.trianglelist = (int *) NULL;          
  mid.edgelist = (int *) NULL;             
  mid.edgemarkerlist = (int *) NULL;   

  
  /* Triangulate the points.  */

  triangulate("pq30zPea0.001", &in, &mid, NULL);


  printf("POINTS: %d\n",mid.numberofpoints);
  for(i=0;i<mid.numberofpoints*2;i+=2)
    printf("%d x %lf y %lf    m %d\n",i/2,mid.pointlist[i],mid.pointlist[i+1],mid.pointmarkerlist[i/2]);
  printf("ELEMS: %d\n",mid.numberoftriangles);
  for(i=0;i<mid.numberoftriangles*3;i+=3)
    printf("%d     %d %d %d\n",i/3,mid.trianglelist[i],mid.trianglelist[i+1],mid.trianglelist[i+2]);
  

 
  /* Free all allocated arrays, including those allocated by Triangle. */

  free(mid.pointlist);
  free(mid.pointmarkerlist);
  free(mid.trianglelist);
  free(mid.edgelist);
  free(mid.edgemarkerlist);

  return 0;
}
