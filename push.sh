#!/bin/sh
/usr/bin/git add -A
/usr/bin/git commit -m "`date +%Y-%m-%d,%H:%m`" 
/usr/bin/git push -u origin master -f


