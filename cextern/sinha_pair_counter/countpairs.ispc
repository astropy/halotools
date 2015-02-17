#ifdef DOUBLE_PREC
#define DOUBLE double
#else
#define DOUBLE float
#endif

export void compute_distances_two_cells(uniform const int N1,
	   									uniform const DOUBLE uniform x1[],
										uniform const DOUBLE uniform y1[],						
										uniform const DOUBLE uniform z1[],
										uniform const int N2,
	   									uniform const DOUBLE uniform x2[],
										uniform const DOUBLE uniform y2[],						
										uniform const DOUBLE uniform z2[],
										uniform const int Nbins,
										uniform const DOUBLE rupp_sqr[],
										uniform unsigned int npairs[])
{
	const uniform DOUBLE rpmin_sqr = rupp_sqr[0];
	const uniform DOUBLE rpmax_sqr = rupp_sqr[Nbins-1];
	const uniform DOUBLE dlogrp = (0.5*(log(rpmax_sqr)-log(rpmin_sqr)))/(Nbins-2);
	const uniform DOUBLE inv_dlogrp = 1.0/dlogrp;
	const uniform DOUBLE logrpmin = log(rpmin_sqr);
	foreach(i=0...N1,j=0...N2) {
		DOUBLE dx = x1[i] - x2[j];
		DOUBLE dy = y1[i] - y2[j];
		DOUBLE dz = z1[i] - z2[j];
		DOUBLE r2 = dx*dx + dy*dy + dz*dz;
		varying int kbin=Nbins;
		cif(r2 >= rpmax_sqr || r2 < rpmin_sqr) {
			kbin = Nbins;
		} else {
			kbin = (int) (0.5*log(r2) - logrpmin)*inv_dlogrp;
		}

		foreach_active(instanceNum) {
			uniform int unifIndex = extract(kbin, instanceNum);
			/* print("kbin = % unifIndex = % programIndex=% \n",kbin,unifIndex,programIndex); */
			//npairs[unifIndex]++;
			atomic_add_global(&npairs[unifIndex], (varying int32) 1);
		}
	}
}
