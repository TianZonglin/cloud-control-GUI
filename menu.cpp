#include "include/menu.h"
#include "include/vis.h"
#include "include/io.h"


void Menu::display()										//Run the visualization associated to the currently-selected
{															//menu entry, if any
 if (current>=menu.size())
 {
 }
 else 
 {
    const int menu_x=20,menu_y=20;
	MenuEntry& item = menu[current];
	if (item.callback) item.callback(dpy);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity(); 
	gluOrtho2D(0.0, dpy.winSize, 0.0, dpy.winSize); 
	
	char buf[300];
	sprintf(buf,"%s",item.menu_name);

	glColor3f(0,0,0);
	glRasterPos2f(menu_x,dpy.winSize-menu_y);
	glutDrawString(buf);								 
	glColor3f(1,1,1);
	glRasterPos2f(menu_x+1,dpy.winSize-menu_y+1);
	glutDrawString(buf);								 

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
 }
}	



void Menu::add(SHOW_WHAT sw,const char* nm,MenuEntry::Callback cb)
{
	menu.push_back(MenuEntry(sw,nm,cb));
}


