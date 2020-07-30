 


```c
this->load(filename); ////读取颜色表
printf("aaaaaa");

void Colormap::load(std::string filename) {
    
    ... strange code here ...

    printf("bbbbbbbb\n");
}   

```

ouptput


```
bbbbbbbb
segmentation fault (core dump)
```

the question is this code just output `bbbbbbbb` but not `aaaaaa`