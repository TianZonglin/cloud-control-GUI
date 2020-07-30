#!/bin/sh
git add -A
git commit -m "`date +%Y-%m-%d,%H:%m`" 
git push -u origin master -f


