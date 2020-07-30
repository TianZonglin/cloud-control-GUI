#include "utils.h"
#include <stdio.h>
#include <stdlib.h>


int file_no_lines(const char* fname)                    //Get #lines of text in text file 'fname'
{
    
    FILE *fp;
    char path[1035];
    char cmd[1024];
    
    sprintf(cmd,"wc -l %s",fname);                      //Synthesize shell command to call to count lines
    
    int ret = -1;                                       //By default, we don't have a valid result
    fp = popen(cmd, "r");                              //Call shell command, pipe its stdout to fp
    if (fp)
    {
        if (fgets(path, sizeof(path)-1, fp))            //Can read at least a line from stdout?
        {
            sscanf(path,"%d",&ret);                     //If so, the #lines is the first string on this line
        }
    }
    
    pclose(fp);                                         //Close pipe to command's stdout
    
    return ret;                                         //Return obtained #lines or -1 if error
}

