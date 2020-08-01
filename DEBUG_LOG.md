 


```c
this->load(filename); ////读取颜色表
printf("aaaaaa");

void Colormap::load(std::string filename) {
    
    ... strange code here ...

    printf("bbbbbbbb\n");
}   

```

ouptput


```c
bbbbbbbb
segmentation fault (core dump)
```

the question is this code just output `bbbbbbbb` but not `aaaaaa`




```c
----int main(int, char**)
--------void loadParameters(int, char**)
----after loadParameters. pointfile = DATA/winequality
--------------------------------void Colormap::load(int)
--------------------------------void Colormap::load(std::__cxx11::string)
--------------------------------load, selected map is colormaps/l-heated-object.cmap
----test colorMap.load, get(1) = (1.000000,0.682353,0.000000)
----not concerned about skelft2DInitialization()
--------PointCloud* loadPointCloud()
--------loadPointCloud, Reading PEx file (loadPointCloud)...
------------bool PointCloud::myLoadPex(const char*, const char*, bool)
------------myLoadPex, Reading 2D points file: DATA/winequality.lamp.2d
------------myLoadPex, Reading projection-errors file: DATA/winequality.lamp.2d
------------Warning: Errors file DATA/winequality.lamp.err doesn't exist
------------myLoadPex, Reading nD points file: DATA/winequality.data
------------attributes_min=3.800000
------------attributes_min=0.080000
------------attributes_min=0.000000
------------attributes_min=0.600000
------------attributes_min=0.009000
------------myLoadPex, Computing squared distance matrix... 
--------loadPointCloud, Finished reading PEx file.
----after, X[-0.527638,-0.644383], Y[0.761702,0.912482]
--------void PointCloud::myFitToSize(float, float, float, float)
----after, X[81.919998,81.919998], Y[942.080017,942.080017]
--------void PointCloud::initEnd()
--------siteDT generated.
--------Triangulating cloud...
----------------void PointCloud::triangulate()
----------------Call Triangle-lib, edges = 15939, tris = 10622
----------------SparseMatrix generated
----------------void PointCloud::makeKDT()
--------Making pointcloud KDT finished
----------------void PointCloud::sortErrors()
--------Computing global variance...
----------------void PointCloud::computeDiameter()
--------Cloud diameter: 936.776
--------Cloud bounding box: 1216.45
--------initEnd finished
--------Display::Display(int, int, PointCloud*, int, char**)
--------------------------------void Colormap::load(int)
------------Grouping* PointCloud::groupByLabel()
------------void PointCloud::dimensionRankCentroid()
------------Centroid ranking complete
------------void PointCloud::dimensionRank(float)
---------------Loop dimensionRankAvg()
----------------Loop dimRankContributionAvg()
------------dimensionRank finished...
------------void Display::computeDimensionRankMap()
----------------void PointCloud::filterRankings(bool)
----------------dimrank_topdims = empty
--------------------void PointCloud::computeTopDims()
--------------------------------void Colormap::load(int)
--------------------------------void Colormap::load(std::__cxx11::string)
--------------------------------load, selected map is colormaps/categorical.cmap
--------------------dimHistogram[0].frequency=4323
--------------------dimHistogram[1].frequency=2172
--------------------dimHistogram[2].frequency=0
--------------------dimHistogram[3].frequency=0
--------------------dimHistogram[4].frequency=0
--------------------dimHistogram[5].frequency=0
--------------------dimHistogram[6].frequency=0
--------------------dimHistogram[7].frequency=0
--------------------------------void Colormap::load(int)
--------------------------------void Colormap::load(std::__cxx11::string)
--------------------------------load, selected map is colormaps/l-heated-object.cmap
----------------void ExplanatoryMap::buildTopRankedNew()
--------------------------------void Colormap::load(int)
--------------------------------void Colormap::load(std::__cxx11::string)
--------------------------------load, selected map is colormaps/categorical.cmap
--------------------float* ExplanatoryMap::buildWithoutShepard(Scalar&, Scalar&)
--------------------------------void Colormap::load(int)
--------------------------------void Colormap::load(std::__cxx11::string)
--------------------------------load, selected map is colormaps/l-heated-object.cmap
------------void Display::displayCb()
----------------void Display::drawMap()
--------------------void draw_dimension_ranking(Display&)
------------------------void draw_image(Display&, int)
----------------------------Finished
```









