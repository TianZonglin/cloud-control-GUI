#include <math.h>
#include <algorithm>
#include <fstream>
#include "include/glutwrapper.h" 
#include "include/io.h"
#include "include/pointcloud.h"
#include "include/config.h"


using namespace std;


int Colormap::getSize() {
    return colorTable.size();
}

void Colormap::setColorMode(bool isColorMode) {
    this->colorMode = isColorMode;
}

bool Colormap::isColorMode() {
    return this->colorMode;
}



void Colormap::load(int mapCode) {
    cout<<"\n"<<"--------------------------------"<<__PRETTY_FUNCTION__<<endl;
    
    if ((loaded_map_code == mapCode) && (inverted == invert_colormap)){
        cout<<"\n"<<"--------------------------------return"<<endl;
        return;
    }
        
      
    colorTable.clear();
    std::string filename = "colormaps/";
    filename += CMAPFILES[mapCode];
    filename += ".cmap";
    //printf("/////1/////");
    this->load(filename); ////读取颜色表
    //printf("/////2/////");
    //printf("\n$$$ after load colormap, colorTable(1) = (%f,%f,%f)\n",colorTable[1].r,colorTable[1].g,colorTable[1].b);

    loaded_map_code = mapCode;
    inverted = invert_colormap;
    current_cmap = mapCode;
    this->name = filename;
}

void Colormap::load(std::string filename) {
    cout<<"\n"<<"--------------------------------"<<__PRETTY_FUNCTION__<<endl;
    ifstream cmapfile(filename.c_str());
    printf("\n--------------------------------load,selected map is %s\n",filename.c_str());
    string line;
    int i = 0;
    int numColors;
    bool colors_read = false;
    int r, g, b;
    while (getline(cmapfile, line)) {
        if (line.c_str()[0] != '#') { // ignore comments
            if (!colors_read) {
                numColors = atoi(line.c_str()); /////第一行是9,即行数,第二行才是数值
                colors_read = true;
                if (invert_colormap) {
                    i = numColors - 1;
                }
                //printf("@%d@",numColors);
            }
            else { /////////error1: read wrong, colorTable has wrong items
                    /////////solved: should use %f to print instead of %d
                sscanf(line.c_str(), "%d %d %d\n", &r, &g, &b); 
                
                //printf("\n%d %d %d\n", r, g, b);
             
                Color currentColor(r, g, b); ////颜色结构体
                //printf("\n$$$ currentColor = (%f,%f,%f)\n",currentColor.r,currentColor.g,currentColor.b);
                //printf("%d",invert_colormap);
                if (invert_colormap) {
                    //printf("<");
                    colorTable.insert(colorTable.begin(), currentColor);
                    //printf(">");
                }
                else {
                    //printf("<");
                    colorTable.push_back(currentColor);
                    //printf(">");
                }
                
            }
        }
    }/**/

    
    //printf("\n");
    //for(int x=0;x<colorTable.size();x++){
     //   printf("(%f,%f,%f)\n",colorTable.at(x).r,colorTable.at(x).g,colorTable.at(x).b);
    //}  正常
 
}    


void rgb2hsv(float r, float g, float b, float& h, float& s, float& v) {
    //pout("------------------------");
    float min = std::min(b,std::min(r,g));
    float max = std::max(b,std::max(r, g));   
    float delta = max - min;
    v = max;
    // NOTE: if Max is == 0, this divide would cause a crash
    if( max > 0.0 ) {
        s = (delta / max);
    } else {
        // if max is 0, then r = g = b = 0              
        s = 0.0;
        h = 0.0;
        v = 0.0;
        return;
    }
    
    // > is bogus, just keeps compiler happy
    if( r >= max )
        // between yellow & magenta
        h = (g - b) / delta;        
    else if( g >= max )
        // between cyan & yellow
        h = 2.0 + (b - r) / delta;  
    else
        // between magenta & cyan
        h = 4.0 + (r - g) / delta;  

    // degrees
    h *= 60.0;                              

    if( h < 0.0 )
        h += 360.0;
}
 
/**
 * Given a color in HSV, return its representation in RGB.
 * @param h Hue color input.
 * @param s Saturation color input.
 * @param v Value color input.
 * @param r Red color output.
 * @param g Green color output.
 * @param b Blue color output.
 */
void hsv2rgb(float h, float s, float v, float& r, float &g, float& b)
{
    //pout("------------------------");
    double      hh, p, q, t, ff;
    long        i;

    if(s <= 0.0) {       // < is bogus, just shuts up warnings
        r = v;
        g = v;
        b = v;
        return;
    }
    hh = h;
    
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = v * (1.0 - s);
    q = v * (1.0 - (s * ff));
    t = v * (1.0 - (s * (1.0 - ff)));

    switch(i) {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;

    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
    default:
        r = v;
        g = p;
        b = q;
        break;
    }
}
		

void float2rgb(float value,float& R,float& G,float& B,bool color_or_grayscale)	//simple color-coding routine
{
   if (!color_or_grayscale)
   { 
	 R = G = B = value;
   }	 
   else
   {
     const float dx=0.8f;

     value = (6-2*dx)*value+dx;
     R = max(0.0f,(3-(float)fabs(value-4)-(float)fabs(value-5))/2);
     G = max(0.0f,(4-(float)fabs(value-2)-(float)fabs(value-4))/2);
     B = max(0.0f,(3-(float)fabs(value-1)-(float)fabs(value-2))/2);
   }	 
}

		
void glutDrawString(const char* s)
{
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10,*s++);
}


void glVertex2f(const Point2d& p)
{
	glVertex2f(p.x,p.y);
}

void drawSplat(const Point2d& p,float rad)		//Draw a splat (current texture) on a radius of 'rad' pixels centered at 'c'
{
		glTexCoord2f(0,0);											
		glVertex2f(p.x-rad,p.y-rad);
		glTexCoord2f(1,0);
		glVertex2f(p.x+rad,p.y-rad);
		glTexCoord2f(1,1);
		glVertex2f(p.x+rad,p.y+rad);
		glTexCoord2f(0,1);
		glVertex2f(p.x-rad,p.y+rad);
}

void setTexture(GLuint tex_id,bool tex_interp)
{
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex_id);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (tex_interp)?GL_LINEAR:GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (tex_interp)?GL_LINEAR:GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

void drawCircle(const Point2d& c,float rad)
{
	const int N = 30;
	
	glBegin(GL_LINE_LOOP);
	for(int i=0;i<N;++i)
	{
	   float alpha = 2*M_PI*float(i)/N;	
	   float x = rad*cos(alpha);
	   float y = rad*sin(alpha);
	   glVertex2f(c.x+x,c.y+y);
	}	
	glEnd();
	
}



Color Colormap::getColor(float value) {

    int numColors = colorTable.size() - 1;
    Color color(0.0f, 0.0f, 0.0f);
    
    if (isnan(value)) 
        return color;
    
    //If not color scale, should be gray scale. Every component have the same value 
    if (!colorMode)
    {
        color.setColor(value, value, value);
        return color;
    }
  
    const float dx=0.8f;
    switch (current_cmap)
    {        
        case CMAP_RAINBOW:
            value = (6-2*dx)*value+dx;
            color.r = max(0.0f,(3-(float)fabs(value-4)-(float)fabs(value-5))/2);
            color.g = max(0.0f,(4-(float)fabs(value-2)-(float)fabs(value-4))/2);
            color.b = max(0.0f,(3-(float)fabs(value-1)-(float)fabs(value-2))/2);
            break;
        case CMAP_CATEGORICAL: {
            int colorIndex = (int)value;
            if (colorIndex >= getSize())
                colorIndex = getSize() - 1;
            color = getColor(colorIndex);
            break;
        }
        default:
            ///checkLoadColormap(current_cmap);
            load(current_cmap);
            float index = value * ((float)numColors);
            int index_i = (int)(index);
            float delta = index - ((float)index_i);
            color.r = (float)colorTable[index_i].r;
            color.g = (float)colorTable[index_i].g;
            color.b = (float)colorTable[index_i].b;
            if (index_i < (numColors))
            {
                color.r += (((float)(colorTable[index_i + 1].r - colorTable[index_i].r)) * delta);
                color.g += (((float)(colorTable[index_i + 1].g - colorTable[index_i].g)) * delta);
                color.b += (((float)(colorTable[index_i + 1].b - colorTable[index_i].b)) * delta);
            }                      
            break;
    }
    
    return color;
}



Color Colormap::getColor(int index) {
    if (index < colorTable.size())
        return colorTable[index];
    else
        return colorTable[colorTable.size()-1];
}







Colormap colorMap;


