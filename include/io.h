#pragma once


#include "include/glutwrapper.h" 
#include <iostream>
#include <vector>


#define CMAP_RAINBOW   0
#define CMAP_HEATED    5
#define CMAP_CATEGORICAL 3


class Point2d;

void			float2rgb(float value,float& r,float& g,float& b,bool color_or_grayscale);
void			glutDrawString(const char* s);
void			glVertex2f(const Point2d&);
void			drawSplat(const Point2d& p,float rad);		//Draw a splat (current texture) on a radius of 'rad' pixels centered at 'c'
void			setTexture(GLuint tex_id,bool tex_interp);
void			drawCircle(const Point2d& c,float rad);

//General-purpose color
struct Color
{
    float r,g,b;
    Color() { r=g=b=1; }
    Color(float r_,float g_,float b_): r(r_),g(g_),b(b_) {}
    Color(int r_, int g_, int b_) : r(r_/255.0f), g(g_/255.0f), b(b_/255.0f) {}
    void setColor(float _r, float _g, float _b) {r = _r; g = _g; b = _b;}
};

static char CMAPFILES [][250] =
{
	"", // rainbow
	"blue-to-cyan",
	"blue-to-yellow",
        "categorical", 
	"l-gray",
	"l-heated-object",
	"l-magenta",
	"locs",
	"l-rainbow",
	"nl-gray",
	"optimal",
	"ryg",
	"ryb",
};

class Colormap {
    
    public:
        Colormap() {
            loaded_map_code = -1;
            colorMode  = true;
            inverted = true;
        }
        void  load(int mapCode);
        Color getColor(int index);
        Color getColor(float value);
        int   getSize();
        void  setColorMode(bool isColorMode);
        bool  isColorMode();
        void  setInverted(bool _inverted);
        bool  isInverted();
        
    private:
        std::string name;
        std::vector<Color> colorTable;
        int   loaded_map_code;
        bool  colorMode;
        bool  inverted;
        void  load(std::string filename);
        
};

extern Colormap colorMap;

void    rgb2hsv(float r, float g, float b, float& h, float& s, float& v);
void    hsv2rgb(float h, float s, float v, float& r, float &g, float& b);



