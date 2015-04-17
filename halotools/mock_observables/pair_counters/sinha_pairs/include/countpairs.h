#pragma once

void countpairs(const int ND1, const double * const X1, const double * const Y1, const double  * const Z1,
				const int ND2, const double * const X2, const double * const Y2, const double  * const Z2,
				const double xmin, const double xmax,
				const double ymin, const double ymax,
				const double zmin, const double zmax,
				const int autocorr,
				const double rpmax,
#ifdef USE_OMP
				const int numthreads,
#endif
				const int nrpbin, const double * restrict rupp, int **paircounts);