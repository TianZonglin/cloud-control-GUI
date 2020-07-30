#pragma once

#include <vector>

class Display;




struct Menu
{
    enum SHOW_WHAT { SHOW_B=0, SHOW_DT, SHOW_AGGR_ERROR, SHOW_REL_ERROR, SHOW_SCALARS, SHOW_END };

	struct MenuEntry
	{
	  typedef void (*Callback)(Display&);
	  SHOW_WHAT	   show_what;
	  const char*  menu_name;
	  Callback     callback;
				   MenuEntry(SHOW_WHAT sw=SHOW_END,const char* nm="",Callback f=0):show_what(sw),menu_name(nm),callback(f) {}
	};
		   Menu(Display& dpy_):current(0),dpy(dpy_) {}
	void   add(SHOW_WHAT,const char*,MenuEntry::Callback);
	void   next()
		   { if (menu.size()) current = (current+1) % menu.size(); }
	void   set(SHOW_WHAT sw)
		   {
		     for(int i=0;i<menu.size();++i)
			    if (menu[i].show_what==sw)
				{ current = i; break; } 	   
		   }		
	void   display();		

int						current;	
std::vector<MenuEntry>	menu;	    
Display&				dpy;
};




