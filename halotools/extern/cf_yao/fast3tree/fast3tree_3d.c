#include <inttypes.h>
struct mytype{int64_t idx; float pos[3];};
#define FAST3TREE_TYPE struct mytype
#define FAST3TREE_DIM 3
#include "fast3tree.c"
