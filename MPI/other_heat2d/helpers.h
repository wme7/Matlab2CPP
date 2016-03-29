typedef struct {
    float* grid;
    float diff;
} result;

void initialize(float** grid, int n, int height, int numProcs, int myID);
void print(float** grid, int n, int height, int myID);
void print_buffer(float* buf, int length, int step);
void handle_mpi_error(int ierr);
