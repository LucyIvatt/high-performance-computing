#define INDEX_2D(i, j, m) ((i*m+j))

struct DoubleArray2D {
    double* data;
    int size_x;
    int size_y;
};

struct CharArray2D {
    char* data;
    int size_x;
    int size_y;
};

