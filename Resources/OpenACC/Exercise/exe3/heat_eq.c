/*
 * SERIAL VERSION
 * Solving conduction of tempreature in a metal plate 
 * modelled by 2D Laplace equation. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef VIS
#include "vis.h"
#endif

#ifndef ROWS
#define ROWS		1000
#endif
#ifndef COLS
#define COLS		1000
#endif
#ifndef MAX_ITER
#define MAX_ITER	1000
#endif
#ifndef TOL
#define TOL		1e-2
#endif
#ifndef MAX_TEMP
#define MAX_TEMP	100.0
#endif
#ifndef CHKPNT_ITER
#define CHKPNT_ITER	100
#endif


void init_grid();

// Allocated 2D stecil. T_old is a copy of T_new from previous iteration
float T_new[ROWS+2][COLS+2], 
      T_old[ROWS+2][COLS+2];
int main(int argc, char* argv[]){

	int max_iter = MAX_ITER,
	    iter = 0;


	// Maximum tempreature change 
	float dT = MAX_TEMP;

	if (argc > 1){
		for (int i=0; i < argc; i++){
			if (argv[i][0] == '-') {
				if (argv[i][1] == 'm'){
					max_iter = atoi(argv[i+1]);
				}
			}
		}
	}
	//HINT: Use data construct to copy of data one off to device. Mind the scoping.

	
		// Initialize the stencil
		init_grid();




		while ( dT > TOL && iter <= max_iter ){
			int i=0,j=0;

			// Optional: if we wanted to write our results to a file 
			// (use make VIS=1 to compile with I/O support)
#ifdef VIS
			if ( iter % CHKPNT_ITER == 0){

				write_dataset_vtk(ROWS+2,COLS+2,T_new,iter);

			}
#endif


			// Evaluate tempreature on interior grid points
			// Hint: make use of kernels directive
#pragma acc kernels 
			for (i=1 ; i <=ROWS; i++){
				for(j=1; j <=COLS; j++){
					T_new[i][j] = 0.25 * (T_old[i-1][j] + T_old[i+1][j] + T_old[i][j-1] + T_old[i][j+1]); 
				}
			}

			dT = 0.0;
			// Evaluate maximum tempreature change and update the T_old with T_new for next iteration
			// Hint: make use of kernerls directive
#pragma acc kernels 
			for (i=1 ; i <=ROWS; i++){
				for(j=1; j <COLS; j++){
					dT = fmaxf(fabsf(T_new[i][j] - T_old[i][j]),dT);
					T_old[i][j] = T_new[i][j];
				}
			}
			iter++;
		}


	printf ( "Stencil size: %d by %d\nConverged in %d iterations with an error of %2.4f\n",ROWS,COLS,iter-1,dT);
	return 0;
}

void init_grid(){
	int i=0,j=0;
	//Hint: Since init_grid is called on Device, a kernel must also be generated for these loops 
		
	
		for (i = 0; i < ROWS+2; i++){
			for (j = 0; j < COLS+2; j++){
				T_old[i][j] = 0.0;
				T_new[i][j] = 0.0;
			}
		}

		// Set top boundary initial values
		for (j = 0; j < COLS+2; j++){
			T_old[0][j] = MAX_TEMP;
			T_new[0][j] = MAX_TEMP;
		}

		// Set left boundary initial values
		for (i = 0; i < ROWS+2; i++){
			T_old[i][0] = MAX_TEMP;
			T_new[i][0] = MAX_TEMP;
		}
}
