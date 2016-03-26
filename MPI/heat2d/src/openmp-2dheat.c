/*
 * B. Estrade <estrabd@cs.uh.edu>
 * Original coding: Spring 2004
 * Serialized: Summer 2010
 * Wrapped into MPI/OpenMP 2dheat Suite: Summer 2010 
 * OpenMP added: ..not yet! :)
 *
 * Serial implementation of 2d heat conduction
 * finite difference over a rectangular domain using:
 * - Jacobi
 * - Gauss-Seidel
 * - SOR
 *
 * This code was created by eliminating the MPI include and 
 * API function calls from the parallel version in order to 
 * serve as a starting point for inserting OpenMP directives.
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       T_SRC0 @ X - (W/2,H)
          |
  *******X*******
  *.............*      
  *.............*	
  *.............*	
  *.............*	
  *.............* ~ 0.0 @ all bdy by "X" (W/2,H)	
  *.............*	
  *.............*	
  *.............*	
  *.............*	
  *.............*	
  ***************
  
  2D domain - WIDTH x HEIGHT
  "X" = T_SRC0
  "*" = 0.0
  "." = internal node suceptible to heating  
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
 */

#define _WIDTH   50 
#define _HEIGHT  50 
#define H        1.0   
#define _EPSILON 0.1 
/* 
  methods:
  1 - jacobi
  2 - gauss-seidel
  3 - sor
*/
#define _METHOD 2 
#define ITERMAX 10    
#define T_SRC0  550.0 
#define ROOT    0

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h> 
#include <stdint.h>
#include <sys/time.h>
#include <time.h>


/* declare functions */
int get_start             (int rank);
int get_end               (int rank);
int get_num_rows          (int rank);
void init_domain          (float ** domain_ptr,int rank);
void jacobi               (float ** current_ptr,float ** next_ptr);
void gauss_seidel         (float ** current_ptr,float ** next_ptr);
void sor                  (float ** current_ptr,float ** next_ptr);
float get_val_par         (float * above_ptr,float ** domain_ptr,float * below_ptr,int rank,int i,int j);
void enforce_bc_par       (float ** domain_ptr,int rank,int i,int j);
int global_to_local       (int rank, int row);
float f                   (int i,int j);
float get_convergence_sqd (float ** current_ptr,float ** next_ptr,int rank);

/* declare and set globals */
int WIDTH=_WIDTH;
int HEIGHT=_HEIGHT;
int meth=_METHOD;
int num_threads;
float EPSILON=_EPSILON;

/* Function pointer to solver method of choice */
void (*method) ();

int main(int argc, char** argv) {
  int p,my_rank,time;
  /* arrays used to contain each PE's rows - specify cols, no need to spec rows */
  float **U_Curr;    
  float **U_Next;
  /* helper variables */
  float convergence,convergence_sqd,local_convergence_sqd;
  /* available iterator  */
  int i,j,k,m,n; 
  int per_proc,remainder,my_start_row,my_end_row,my_num_rows;
  int verbose = 0;
  int show_time = 0;
  /* for timings */
  struct timeval tv;
  struct timezone tz;
  struct tm *tm;
  
  /* artifacts of original serialization from MPI version */
  p = 1;
  my_rank = 0;

  /* argument processing done by everyone */
  int c,errflg;
  extern char *optarg;
  extern int optind, optopt;

  while ((c = getopt(argc, argv, "e:h:m:tw:v")) != -1) {
      switch(c) {
      case 'e':
          EPSILON = atof(optarg);
          break;
      case 'h':
          HEIGHT = atoi(optarg);
          break;
      case 'm': 
          /* selects the numerical methods */
          switch(atoi(optarg)) {
          case 1: /* jacobi */
            meth = 1; 
            break;
          case 2: /* gauss-seidel */
            meth = 2;
            break;
          case 3: /* sor */
            meth = 3;
            break;
          } 
          break;
      case 't': 
          show_time++; /* overridden by -v (verbose) */
          break;
      case 'w':
          WIDTH = atoi(optarg);
          break;
      case 'v':
          verbose++;
          break;
     /* handle bad arguments */
      case ':': /* -h or -w without operand */
          if (ROOT == my_rank)
              fprintf(stderr,"Option -%c requires an operand\n", optopt);
          errflg++;
          break;
      case '?':
          if (ROOT == my_rank)
              fprintf(stderr,"Unrecognized option: -%c\n", optopt);
          errflg++;
          break;
      }
  }

/*
  if (0 < errflg) 
      exit(EXIT_FAILURE);
*/

  /* wait for user to input runtime params */
  //MPI_Barrier(MPI_COMM_WORLD);

  /* broadcast method to use  */
  //(void) MPI_Bcast(&meth,1,MPI_INT,0,MPI_COMM_WORLD);

  switch (meth) {
    case 1:
      method = &jacobi;
    break;
    case 2:
      method = &gauss_seidel;
    break;
    case 3:
      method = &sor;
    break;
  }  
   
  /* let each processor decide what rows(s) it owns */
  my_start_row = get_start(my_rank);
  my_end_row = get_end(my_rank);
  my_num_rows = get_num_rows(my_rank);

  if ( 0 < verbose )
    printf("proc %d contains (%d) rows %d to %d\n",my_rank,my_num_rows,my_start_row,my_end_row);
  fflush(stdout);  

  /* allocate 2d array */
  U_Curr = (float**)malloc(sizeof(float*)*my_num_rows);
  U_Curr[0] = (float*)malloc(sizeof(float)*my_num_rows*(int)floor(WIDTH/H));
  for (i=1;i<my_num_rows;i++) {
    U_Curr[i] = U_Curr[i-1]+(int)floor(WIDTH/H);
  }

  /* allocate 2d array */
  U_Next = (float**)malloc(sizeof(float*)*my_num_rows);
  U_Next[0] = (float*)malloc(sizeof(float)*my_num_rows*(int)floor(WIDTH/H));
  for (i=1;i<my_num_rows;i++) {
    U_Next[i] = U_Next[i-1]+(int)floor(WIDTH/H);
  }
  
  /* initialize global grid */
  init_domain(U_Curr,my_rank);
  init_domain(U_Next,my_rank);
  
  /* iterate for solution */
  if (my_rank == ROOT) {
    gettimeofday(&tv, &tz);
    tm=localtime(&tv.tv_sec);
    time = 1000000*(tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec) + tv.tv_usec; 
  }
  k = 1;

  num_threads = 0;    
  //while (1) {
  for(;;) {
    method(U_Curr,U_Next);

    local_convergence_sqd = get_convergence_sqd(U_Curr,U_Next,my_rank);
    //MPI_Reduce(&local_convergence_sqd,&convergence_sqd,1,MPI_FLOAT,MPI_SUM,ROOT,MPI_COMM_WORLD);
    convergence_sqd = local_convergence_sqd;
    if (my_rank == ROOT) {
      convergence = sqrt(convergence_sqd);
      if (verbose == 1) {
        printf("L2 = %f\n",convergence);
        fflush(stdout);
      }
    } 

    /* broadcast method to use */
    //(void) MPI_Bcast(&convergence,1,MPI_INT,0,MPI_COMM_WORLD);
    if (convergence <= EPSILON) {
      break;      
    }
    
    /* copy U_Next to U_Curr */
    for (j=my_start_row;j<=my_end_row;j++) {
      for (i=0;i<(int)floor(WIDTH/H);i++) {
        U_Curr[j-my_start_row][i] = U_Next[j-my_start_row][i];
      }
    }
    k++;
    //MPI_Barrier(MPI_COMM_WORLD);    
  }   

  /* say something at the end */
  if (my_rank == ROOT) {
    gettimeofday(&tv, &tz);
    tm=localtime(&tv.tv_sec);
    time = 1000000*(tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec) + tv.tv_usec - time; 
    if (0 < verbose)
    {   printf("Estimated time to convergence in %d iterations using %d processors on a %dx%d grid is %d microseconds\n",k,p,(int)floor(WIDTH/H),(int)floor(HEIGHT/H),time);
    } 
    else if (show_time)  
    {   printf("% 5d\t% 12d msec\n",omp_get_max_threads(),time); }

    /* else show nothing */
  }

  exit(EXIT_SUCCESS);
  //return 0; not needed; not reached
}

 /* used by each PE to compute the sum of the squared diffs between current iteration and previous */

 float get_convergence_sqd (float ** current_ptr,float ** next_ptr,int rank) {
    int i,j,my_start,my_end,my_num_rows;
    float sum;
    
    my_start = get_start(rank);
    my_end = get_end(rank);
    my_num_rows = get_num_rows(rank);
 
    sum = 0.0;
    for (j=my_start;j<=my_end;j++) {
      for (i=0;i<(int)floor(WIDTH/H);i++) {
        sum += pow(next_ptr[global_to_local(rank,j)][i]-current_ptr[global_to_local(rank,j)][i],2);
      }
    }
    return sum;   
 }

 /* implements parallel jacobi methods */

 void jacobi (float ** current_ptr,float ** next_ptr) {
    int i,j,p,my_rank,my_start,my_end,my_num_rows;
    float U_Curr_Above[(int)floor(WIDTH/H)];  /* 1d array holding values from bottom row of PE above */
    float U_Curr_Below[(int)floor(WIDTH/H)];  /* 1d array holding values from top row of PE below */

    p = 1;
    my_rank = 0;

    my_start = get_start(my_rank);
    my_end = get_end(my_rank);
    my_num_rows = get_num_rows(my_rank);

#pragma omp parallel default(none) private(i,j) \
  shared(p,my_rank,U_Curr_Above,U_Curr_Below,WIDTH,my_start,my_end,next_ptr,current_ptr)
{
    /* Jacobi method using global addressing */
#pragma omp for schedule(runtime) 
    for (j=my_start;j<=my_end;j++) {
      for (i=0;i<(int)floor(WIDTH/H);i++) {
	next_ptr[j-my_start][i] = .25*(get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i-1,j)
                   + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i+1,j)
                   + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j-1)
                   + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j+1)
                   - (pow(H,2)*f(i,j)));		   
	enforce_bc_par(next_ptr,my_rank,i,j);	   
      }
    }
} //end omp parallel region 

 }

 /* implements parallel g-s method */

 void gauss_seidel (float ** current_ptr,float ** next_ptr) {
    int i,j,p,my_rank,my_start,my_end,my_num_rows;
    float U_Curr_Above[(int)floor(WIDTH/H)];  /* 1d array holding values from bottom row of PE above */
    float U_Curr_Below[(int)floor(WIDTH/H)];  /* 1d array holding values from top row of PE below */
    float W =  1.0;

    p = 1;
    my_rank = 0;

    my_start = get_start(my_rank);
    my_end = get_end(my_rank);
    my_num_rows = get_num_rows(my_rank);
        
#pragma omp parallel default(none) private(i,j) \
  shared(W,p,my_rank,U_Curr_Above,U_Curr_Below,WIDTH,my_start,my_end,next_ptr,current_ptr)
{
    /* solve next reds (i+j odd) */
#pragma omp for schedule(runtime) 
    for (j=my_start;j<=my_end;j++) {
      for (i=0;i<(int)floor(WIDTH/H);i++) {
        if ((i+j)%2 != 0) {
          next_ptr[j-my_start][i] = get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j)
                     + (W/4)*(get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i-1,j)
                     + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i+1,j)
                     + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j-1)
                     + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j+1)
		     - 4*(get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j))
                     - (pow(H,2)*f(i,j))); 
	  enforce_bc_par(next_ptr,my_rank,i,j);	   
        }
      }
    }   
   /* solve next blacks (i+j) even .... using next reds */
#pragma omp for schedule(runtime) 
    for (j=my_start;j<=my_end;j++) {
      for (i=0;i<(int)floor(WIDTH/H);i++) {
        if ((i+j)%2 == 0) {
          next_ptr[j-my_start][i] = get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j)
                     + (W/4)*(get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i-1,j)
                     + get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i+1,j)
                     + get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i,j-1)
                     + get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i,j+1)
                     - 4*(get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i,j))
                     - (pow(H,2)*f(i,j)));
 	  enforce_bc_par(next_ptr,my_rank,i,j);	   
        }
      }
    }     
} //end omp parallel region 
 }
 
 /* implements parallels sor method */
 
 void sor (float ** current_ptr,float ** next_ptr) {
    int i,j,p,my_rank,my_start,my_end,my_num_rows;
    float U_Curr_Above[(int)floor(WIDTH/H)];  /* 1d array holding values from bottom row of PE above */
    float U_Curr_Below[(int)floor(WIDTH/H)];  /* 1d array holding values from top row of PE below */
    float W =  1.5;
    
    p = 1;
    my_rank = 0;

    my_start = get_start(my_rank);
    my_end = get_end(my_rank);
    my_num_rows = get_num_rows(my_rank);
        
#pragma omp parallel default(none) private(i,j) \
  shared(W,p,my_rank,U_Curr_Above,U_Curr_Below,WIDTH,my_start,my_end,next_ptr,current_ptr)
{
#pragma omp for schedule(runtime) 
    /* solve next reds (i+j odd) */
    for (j=my_start;j<=my_end;j++) {
      for (i=0;i<(int)floor(WIDTH/H);i++) {
        if ((i+j)%2 != 0) {
          next_ptr[j-my_start][i] = get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j)
                     + (W/4)*(get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i-1,j)
                     + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i+1,j)
                     + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j-1)
                     + get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j+1)
		     - 4*(get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j))
                     - (pow(H,2)*f(i,j))); 
 	  enforce_bc_par(next_ptr,my_rank,i,j);	   
        }
      }
    }   
   /* solve next blacks (i+j) even .... using next reds */
#pragma omp for schedule(runtime) 
    for (j=my_start;j<=my_end;j++) {
      for (i=0;i<(int)floor(WIDTH/H);i++) {
        if ((i+j)%2 == 0) {
          next_ptr[j-my_start][i] = get_val_par(U_Curr_Above,current_ptr,U_Curr_Below,my_rank,i,j)
                     + (W/4)*(get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i-1,j)
                     + get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i+1,j)
                     + get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i,j-1)
                     + get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i,j+1)
                     - 4*(get_val_par(U_Curr_Above,next_ptr,U_Curr_Below,my_rank,i,j))
                     - (pow(H,2)*f(i,j)));
 	  enforce_bc_par(next_ptr,my_rank,i,j);	   
        }
      }
    }     
} //end omp parallel region 
 }
 
 /* enforces bcs in in serial and parallel */
 
 void enforce_bc_par (float ** domain_ptr,int rank,int i,int j) {
   /* enforce bc's first */
   if(i == ((int)floor(WIDTH/H/2)-1) && j == 0) {
     /* This is the heat source location */
     domain_ptr[j][i] = T_SRC0;
   } else if (i <= 0 || j <= 0 || i >= ((int)floor(WIDTH/H)-1) || j >= ((int)floor(HEIGHT/H)-1)) {
     /* All edges and beyond are set to 0.0 */
     domain_ptr[global_to_local(rank,j)][i] = 0.0;
   }
 }

 /* returns appropriate values for requested i,j */
 
 float get_val_par (float * above_ptr,float ** domain_ptr,float * below_ptr,int rank,int i,int j) {
   float ret_val;
   int p;
      
   /* artifact from original serialization of MPI version */
   p = 1;

   /* enforce bc's first */
   if(i == ((int)floor(WIDTH/H/2)-1) && j == 0) {
     /* This is the heat source location */
     ret_val = T_SRC0;
   } else if (i <= 0 || j <= 0 || i >= ((int)floor(WIDTH/H)-1) || j >= ((int)floor(HEIGHT/H)-1)) {
     /* All edges and beyond are set to 0.0 */
     ret_val = 0.0;
   } else {
     /* Else, return value for matrix supplied or ghost rows */
     if (j < get_start(rank)) {
       if (rank == ROOT) {
         /* not interested in above ghost row */
         ret_val = 0.0;
       } else { 
         ret_val = above_ptr[i];
	 /*printf("%d: Used ghost (%d,%d) row from above = %f\n",rank,i,j,above_ptr[i]);
	 fflush(stdout);*/
       }
     } else if (j > get_end(rank)) {
       if (rank == (p-1)) {
         /* not interested in below ghost row */
         ret_val = 0.0;
       } else { 
         ret_val = below_ptr[i];
	 /*printf("%d: Used ghost (%d,%d) row from below = %f\n",rank,i,j,below_ptr[i]);
	 fflush(stdout);*/
       }     
     } else {
       /* else, return the value in the domain asked for */
       ret_val = domain_ptr[global_to_local(rank,j)][i];
       /*printf("%d: Used real (%d,%d) row from self = %f\n",rank,i,global_to_local(rank,j),domain_ptr[global_to_local(rank,j)][i]);
       fflush(stdout);*/
     }
   }
   return ret_val;

 }

 /* initialized domain to 0.0 - could be where grid file is read in */

 void init_domain (float ** domain_ptr,int rank) {
   int i,j,start,end,rows;
   start = get_start(rank);
   end = get_end(rank);
   rows = get_num_rows(rank);

   for (j=start;j<end;j++) {
     for (i=0;i<(int)floor(WIDTH/H);i++) {
       domain_ptr[j-start][i] = 0.0;
     }
   }
 }

 /* computes start row for given PE */

 int get_start (int rank) {
   /* computer row divisions to each proc */
   int p,per_proc,start_row,remainder;
   /* artifact of serialization of orignal MPI version */
   p = 1; 
   /* get initial whole divisor */
   per_proc = (int)floor(HEIGHT/H)/p;
   /* get number of remaining */
   remainder = (int)floor(HEIGHT/H)%p;
   /* there is a remainder, then it distribute it to the first "remainder" procs */ 
   if (rank < remainder) {
     start_row = rank * (per_proc + 1);
   } else {
     start_row = rank * (per_proc) + remainder;
   }
   return start_row;
 }

 /* computes end row for given PE */

 int get_end (int rank) {
   /* computer row divisions to each proc */
   int p,per_proc,remainder,end_row;
   /* artifact of serialization of orignal MPI version */
   p = 1; 
   per_proc = (int)floor(HEIGHT/H)/p;
   remainder = (int)floor(HEIGHT/H)%p;
   if (rank < remainder) {
     end_row = get_start(rank) + per_proc;
   } else {
     end_row = get_start(rank) + per_proc - 1;
   }
   return end_row;
 }

 /* calcs number of rows for given PE */
 
 int get_num_rows (int rank) {
   return 1 + get_end(rank) - get_start(rank);
 }

 int global_to_local (int rank, int row) {
   return row - get_start(rank);
 }

  /* 
  * f - function that would be non zero if there was an internal heat source
  */
 
 float f (int i,int j) {
   return 0.0;
 }
