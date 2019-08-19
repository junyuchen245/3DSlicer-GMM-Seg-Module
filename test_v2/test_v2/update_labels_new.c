#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "miputil.h"

typedef short int LABELTYPE;
#define INDEX3D(x,y,z, xdim, ydim, zdim) ((xdim)*((z)*(ydim) + (y)) + (x))

//////////////////////////////////////////////////////////////////////////// k means ////////////////////////////////////////////////////////////////////////////
float max(float* voxvals, int nvox)
{
	float max_val = voxvals[0];
	for(int i=0; i<nvox; i++)
	{
		if(voxvals[i]>max_val)
		{
			max_val = voxvals[i];
		}
	}
	return max_val;
}

float min(float* voxvals, int nvox)
{
	float min_val = voxvals[0];
	for(int i=0; i<nvox; i++)
	{
		if(voxvals[i]<min_val)
		{
			min_val = voxvals[i];
		}
	}
	return min_val;
}

int argmin(float* voxvals, int nvox)
{
	float min_val = voxvals[0];
	int min_i = 0;
	for(int i=0; i<nvox; i++)
	{
		if(voxvals[i]<min_val)
		{
			min_val = voxvals[i];
			min_i = i;
		}
	}
	return min_i;
}

int argmin_double(double* voxvals, int nvox)
{
	double min_val = voxvals[0];
	int min_i = 0;
	for(int i=0; i<nvox; i++)
	{
		if(voxvals[i]<min_val)
		{
			min_val = voxvals[i];
			min_i = i;
		}
	}
	return min_i;
}

int argmax(float* voxvals, int nvox)
{
	float max_val = voxvals[0];
	int max_i = 0;
	for(int i=0; i<nvox; i++)
	{
		if(voxvals[i]>max_val)
		{
			max_val = voxvals[i];
			max_i = i;
		}
	}
	return max_i;
}

int argmax_double(double* voxvals, int nvox)
{
	double max_val = voxvals[0];
	int max_i = 0;
	for(int i=0; i<nvox; i++)
	{
		if(voxvals[i]>max_val)
		{
			max_val = voxvals[i];
			max_i = i;
		}
	}
	return max_i;
}

double mean_class(float* voxvals, int nvox, LABELTYPE* label_map, LABELTYPE label)
{
	double sum = 0.0;
	double count = 0;
	for(int i=0; i<nvox; i++)
	{
		if (label_map[i] == label)
		{
			sum += voxvals[i];
			count += 1.0;
		}
	}
	if(count==0)
	return 0;
	return sum/count;
}

double var_class(float* voxvals, int nvox, LABELTYPE* label_map, LABELTYPE label)
{
	double sum = 0.0;
	double count = 0;
	double mean = 0.0;
	double var  = 0.0;
	double nval = 0.0;

	for(int i=0; i<nvox; i++)
	{
		if (label_map[i] == label)
		{
			sum += voxvals[i];
			count += 1.0;
		}
	}
	mean = sum/count;

	sum = 0.0;
	for(int i=0; i<nvox; ++i){
		if (label_map[i] == label){
			sum += pow(mean-voxvals[i],2.0);
		}
	}

	if(count == 0)
	{
		nval = 0.0001;
	}
	else
	{
		nval = (double)count;
	}
	var=sum/nval; //Use the population std dev formula
	//fprintf(stderr, "var is:%f\n", var);
	return var;	
}

void k_means(float* voxvals, LABELTYPE* labels, int xdim, int ydim, int zdim, int num_clus, float delta, int simpleKmeans)
{
	int nvox = xdim*ydim*zdim;
	float part_lb;
	float part_ub;
	part_lb = min(voxvals, nvox); // Max value within the image
	part_ub = max(voxvals, nvox); // Min value within the image
	float ini_means[num_clus];
	for(int i=0; i<num_clus; i++)
	{
		ini_means[i] = ((part_ub - part_lb)/num_clus *((i+1)-0.5) + part_lb);
	}
	
	float means[num_clus];
	//means = &ini_means;
	for(int i=0; i<num_clus; i++)
	{
		means[i] = ini_means[i];
	}
	//fprintf(stderr, "mean1=%f, mean2=%f\n", means[0], means[1]);
	LABELTYPE* label_map;
	LABELTYPE* new_map;
	label_map = (LABELTYPE *)malloc(xdim*ydim*zdim*sizeof(LABELTYPE));
	new_map   = (LABELTYPE *)malloc(xdim*ydim*zdim*sizeof(LABELTYPE));
	int first_enter_flag = 1;
	int max_iter = 40;
	float val = 0.0;
	float pixel_pot[num_clus];
	float penalty = 0.0;
	LABELTYPE label;

	for(int iter=0;iter<max_iter;iter++)    //iteration
	{
		for(int ix=0;ix<xdim;ix++)          //x dim
		{
			for(int iy=0;iy<ydim;iy++)      //y dim
			{
				for(int iz=0;iz<zdim;iz++)  //z dim
				{
					val=voxvals[INDEX3D(ix, iy, iz, xdim, ydim, zdim)];
					for(LABELTYPE k=0;k<num_clus;k++)
					{
						penalty = 0.0;
						if(first_enter_flag == 0 && simpleKmeans == 0)
						{
							for(int ia=-1; ia<2; ia++)
							{
								for(int ib=-1; ib<2; ib++)
								{
									for(int ic=-1; ic<2; ic++)
									{
										if(ix + ia < 0 || iy + ib < 0 || iz + ic < 0 || ix + ia >= xdim || iy + ib >= ydim || iz + ic >= zdim) continue;
										label = label_map[INDEX3D(ix+ia, iy+ib, iz+ic, xdim, ydim, zdim)];
										if(label != k)
										{
											penalty = penalty + 1.0;
										}
									}//ic
								}//ib
							}//ia
						}//if first enter
						float smooth_pot = delta * penalty;
						float ini_pot = (val-means[k])*(val-means[k]);
						pixel_pot[k] = ini_pot + smooth_pot;
					}//class
					new_map[INDEX3D(ix, iy, iz, xdim, ydim, zdim)] = argmin(pixel_pot, num_clus);
				}//z
			}//y
		}//x
		for(int i=0; i < xdim*ydim*zdim; i++) label_map[i] = new_map[i];

		// recompute mean
		for(LABELTYPE k=0; k<num_clus; k++)
		{
			means[k] = mean_class(voxvals, nvox, label_map, k);
		}
		//fprintf(stderr, "k means: mean1=%f, mean2=%f, mean3=%f\n", means[0], means[1], means[2], means[3]);
		first_enter_flag = 0;
	}//iter
	for(LABELTYPE k=0; k<num_clus; k++)
	{
		int max_i;
		max_i = argmax(means, num_clus);
		means[max_i] = -10000.0;
		for(int i=0; i<nvox; i++)
		{
			if(label_map[i]==max_i)
			{
				labels[i] = num_clus - k - 1;
			}
		}
	}
	
	free((void *)new_map);
	free((void *)label_map);
	return;
}
void dot_multiply(double* results, double* vec_1, double* vec_2, int dim)
{
	for(int i=0; i<dim; i++)
	{
		results[i] = vec_1[i] * vec_2[i];
	}
}
void dot_divide(double* results, double* vec_1, double* vec_2, int dim)
{
	for(int i=0; i<dim; i++)
	{
		results[i] = vec_1[i] / vec_2[i];
	}
}
void dot_minus(double* results, double* vec_1, double* vec_2, int dim)
{
	for(int i=0; i<dim; i++)
	{
		results[i] = vec_1[i] - vec_2[i];
	}
}

int MRF_EM(float* voxvals, LABELTYPE* labels, int xdim, int ydim, int zdim, int num_clus, double beta)
{
	LABELTYPE* label_map;
	int nvox = xdim*ydim*zdim; // number of voxels
	double* clique_pot;
	double* clique_pot_2;
	double* ini_pot;
	double* prob;
	double gaus_1;
	double* gaus_2;
	double* gaus_3;
	double* gaus_prob;
	double* mean_cluster;
	double* variance_cluster;
    gaus_2 = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	gaus_3 = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	gaus_prob = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	clique_pot = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	clique_pot_2 = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	ini_pot = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	prob = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	label_map = (LABELTYPE *)malloc(xdim*ydim*zdim*sizeof(LABELTYPE)); // define label map
	mean_cluster=(double *)dvector(num_clus*sizeof(double)); // class means
	variance_cluster=(double *)dvector(num_clus*sizeof(double)); // class variances
	//fprintf(stderr, "class num is:%d\n", num_clus);
	k_means(voxvals, label_map, xdim, ydim, zdim, num_clus, 0, 1); // k means clustering as initialization
	//for(int i=0; i < xdim*ydim*zdim; ++i)
	//	labels[i] = label_map[i];
	//return 0;

	for(int k=0; k<num_clus; k++)
	{
		mean_cluster[k]     = mean_class(voxvals, nvox, label_map, k);
		variance_cluster[k] = var_class(voxvals, nvox, label_map, k);
	}//class k
	//fprintf(stderr, "initial: mean1=%f, mean2=%f, mean3=%f\n", mean_cluster[0], mean_cluster[1], mean_cluster[2]);
	//fprintf(stderr, "initial: var1=%f, var2=%f, var3=%f\n", variance_cluster[0], variance_cluster[1], variance_cluster[2]);
	//fprintf(stderr, "mrf: var1=%f, var2=%f,var3=%f\n", variance_cluster[0], variance_cluster[1], variance_cluster[2]);
	//clique_pot = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	//clique_pot_2 = (double *)malloc(xdim*ydim*zdim*sizeof(double));

	int max_iter = 20;
	int first_enter_flag = 1;
	LABELTYPE label;
	//fprintf(stderr, "beta is:%f\n", beta);
	for(int iter=0; iter<max_iter; iter++)
	{
		//fprintf(stderr, "iteration=%d\n", iter);
		for(int k=0; k<num_clus; k++)
		{
			if(first_enter_flag == 1)
			{
				for(int ix=0;ix<xdim;ix++)          //x dim
				{
					for(int iy=0;iy<ydim;iy++)      //y dim
					{
						for(int iz=0;iz<zdim;iz++)  //z dim
						{
							double tmp_pot = 0.0;
							for(int ia=-1; ia<2; ia++)
							{
								for(int ib=-1; ib<2; ib++)
								{
									for(int ic=-1; ic<2; ic++)
									{
										if(ix + ia < 0 || iy + ib < 0 || iz + ic < 0 || ix + ia >= xdim || iy + ib >= ydim || iz + ic >= zdim) continue;
										if(ia != 0 || ib != 0 || ic != 0)
										{
											label = label_map[INDEX3D(ix+ia, iy+ib, iz+ic, xdim, ydim, zdim)];
											if(k == label)
											{
												tmp_pot -= 1.0;
											}
											else
											{
												tmp_pot += 1.0;
											}
										}
									}// ic
								}// ib
							}// ia
							clique_pot[INDEX3D(ix, iy, iz, xdim, ydim, zdim)+k*nvox] = tmp_pot;
							//if(INDEX3D(ix, iy, iz, xdim, ydim, zdim)%1000 == 0) fprintf(stderr, "tmp_pot1=%f\n", tmp_pot);
						}// iz
					}// iy
				}// ix
			}// if first enter
			else
			{
				for(int clique_i=0; clique_i<nvox; clique_i++)
				{
					clique_pot[clique_i+k*nvox] = clique_pot_2[clique_i+k*nvox];
				}// clique i
			}
			for(int i=0; i<nvox; i++)
			{
				ini_pot[i+k*nvox] = pow(voxvals[i] - mean_cluster[k],2.0)/variance_cluster[k] + log(variance_cluster[k]);
			}

		}//k class
		//fprintf(stderr, "location 0\n");
		double* tmp_pot;
		tmp_pot = (double *)dvector(num_clus*sizeof(double));
		for(int i=0; i<nvox; i++)
		{
			for(int k=0; k<num_clus; k++)
			{
				tmp_pot[k] = ini_pot[i+k*nvox] + beta*clique_pot[i+k*nvox];
/*
				if(i%20000 == 0) 
				{
					//fprintf(stderr, "i = %d\n", i);
					fprintf(stderr, "ini_pot = %f\n", ini_pot[i+k*nvox]);
					fprintf(stderr, "clique_pot = %f\n", clique_pot[i+k*nvox]);
					//fprintf(stderr, "tmp_pot1=%f, tmp_pot2=%f, tmp_pot3=%f\n", tmp_pot[0], tmp_pot[1], tmp_pot[2]);
				}
*/				
			}
			label_map[i] = argmin_double(tmp_pot, num_clus);
			//fprintf(stderr, "tmp_pot=%f\n", tmp_pot[0]);
		}
		//fprintf(stderr, "tmp_pot=%f\n", tmp_pot[0]);
		//fprintf(stderr, "location 1\n");
		free_dvector((void *)tmp_pot);

		double gaus_1;
		////////////////////// EM /////////////////////////////
		for(int k=0; k<num_clus; k++)
		{
			gaus_1 = 1/sqrt(2*3.1415926*sqrt(variance_cluster[k]));
			//fprintf(stderr, "location 2\n");
			for(int i=0; i<nvox; i++)
			{
				gaus_2[i] = -pow((double)voxvals[i]-mean_cluster[k],2.0);
				gaus_3[i] = exp(gaus_2[i]/(2*variance_cluster[k]));
				gaus_prob[i] = gaus_1 * gaus_3[i];
			}
			for(int ix=0;ix<xdim;ix++)          //x dim
			{
				for(int iy=0;iy<ydim;iy++)      //y dim
				{
					for(int iz=0;iz<zdim;iz++)  //z dim
					{
						double tmp_pot = 0.0;
						for(int ia=-1; ia<2; ia++)
						{
							for(int ib=-1; ib<2; ib++)
							{
								for(int ic=-1; ic<2; ic++)
								{

									if(ix + ia < 0 || iy + ib < 0 || iz + ic < 0 || ix + ia >= xdim || iy + ib >= ydim || iz + ic >= zdim) continue;
									if(ia != 0 || ib != 0 || ic != 0)
									{
										label = label_map[INDEX3D(ix+ia, iy+ib, iz+ic, xdim, ydim, zdim)];
										if(k == label)
										{
											tmp_pot -= 1.0;
										}
										else
										{
											tmp_pot += 1.0;
										}
									}
								}// ic
							}// ib
						}// ia
						clique_pot_2[INDEX3D(ix, iy, iz, xdim, ydim, zdim)+k*nvox] = beta*tmp_pot;
					}// iz
				}// iy
			}// ix
			for(int i=0; i<nvox; i++)
			{
				prob[i+k*nvox] = gaus_prob[i]*exp(-clique_pot_2[i+k*nvox]);
				/*
				if(i%20000 == 0) 
				{
					fprintf(stderr, "gaus_prob = %f\n", gaus_prob[i]);
					fprintf(stderr, "-clique_pot_2 = %f\n", -clique_pot_2[i+k*nvox]);
					fprintf(stderr, "exp = %f\n", exp(-clique_pot_2[i+k*nvox]));
				}
				*/
			}

		}// k class
		for(int i=0; i<nvox; i++)
		{
			double tmp_prob=0.0;
			for(int k=0; k<num_clus; k++)
			{
				tmp_prob += prob[i+k*nvox];
			}
			for(int k=0; k<num_clus; k++)
			{
				prob[i+k*nvox] = prob[i+k*nvox]/tmp_prob;
			}
		}

		////////update mean and var//////////
		for(int k=0; k<num_clus; k++)
		{
			double tmp_val_1 = 0.0;
			double tmp_sum   = 0.0;
			double tmp_val_2 = 0.0;
			for(int i=0; i<nvox; i++)
			{
				// for mean
				tmp_val_1 += prob[i+k*nvox]*voxvals[i];
				tmp_sum += prob[i+k*nvox];
			}
			mean_cluster[k] = tmp_val_1/tmp_sum;
			//fprintf(stderr, "tmp_sum=%f\n", tmp_sum);
			for(int i=0; i<nvox; i++)
			{
				// for var
				tmp_val_2 += prob[i+k*nvox]*pow((voxvals[i]-mean_cluster[k]),2.0);
			}
			variance_cluster[k] = tmp_val_2/tmp_sum;
		}
		//fprintf(stderr, "mrf: mean1=%f, mean2=%f,mean3=%f\n", mean_cluster[0], mean_cluster[1], mean_cluster[2]);
		first_enter_flag = 0;

	}//iter
	for(int i=0; i < xdim*ydim*zdim; ++i)
		labels[i] = label_map[i];

	free((void *)clique_pot);
	free((void *)clique_pot_2);
	free((void *)ini_pot);
	free((void *)prob);
	free((void *)gaus_2);
	free((void *)gaus_3);
	free((void *)gaus_prob);
	free((void *)label_map);
	free_dvector((void *)variance_cluster);
	free_dvector((void *)mean_cluster);
	return 0;
}

int* append_int_array(int* input_arr, int size, int num)
{
	int* output_array;
	output_array = (int *)malloc((size+1)*sizeof(int));
	if (size != 0)
	{
		for(int i=0; i<size; i++)
		{
			output_array[i] = input_arr[i];
		}
	}
	output_array[size] = num;
	free((void *)input_arr);
	return output_array;
}


int argmax_int(int* voxvals, int nvox)
{
	int max_val = voxvals[0];
	int max_i = 0;
	for(int i=0; i<nvox; i++)
	{
		if(voxvals[i]>max_val)
		{
			max_val = voxvals[i];
			max_i = i;
		}
	}
	return max_i;
}

int region_grow_cropped3Dimg(LABELTYPE* labeled_roi, int bw_ht, int bw_wid, int bw_len, double seedx, double seedy, double seedz, int num_clus)
{
	int les_reg_label;
	int wid_box    = 1;
	int count      = 0;
	int nvox       = bw_ht*bw_wid*bw_len;
	int count_temp = 0;
	int* x_ord  = NULL;
	int* y_ord  = NULL;
	int* z_ord  = NULL;
	int* temp_arr;
	int x_curr;
	int y_curr;
	int z_curr;
	LABELTYPE* flag;
	LABELTYPE* new_labels;
	flag        = (LABELTYPE *)malloc(nvox*sizeof(LABELTYPE));
	new_labels  = (LABELTYPE *)malloc(nvox*sizeof(LABELTYPE));
	temp_arr    = (int *)dvector(num_clus*sizeof(int));
	
	for(int i=0; i<num_clus; i++)
	{
		temp_arr[i] = 0;
	}
	// initialization
	for(int i=0; i < nvox; ++i)
	{
		flag[i] = 0;
		new_labels[i] = 0;
	}
	seedx = (int)seedx;
	seedy = (int)seedy;
	seedz = (int)seedz;
	
	for(int i=-wid_box; i<=wid_box; i++)
	{
		for(int j=-wid_box; j<=wid_box; j++)
		{
			for(int z=-wid_box; z<=wid_box; z++)
			{
				if(seedx+i >= bw_ht || seedy+j >= bw_wid || seedz+z >= bw_len || seedx+i < 0 || seedy+j < 0 || seedz+z < 0)// check the indexes
                	continue;
				//fprintf(stderr, "check point0\n");
				x_ord = append_int_array(x_ord, count, seedx+i);
				y_ord = append_int_array(y_ord, count, seedy+j);
				z_ord = append_int_array(z_ord, count, seedz+z);
				flag[INDEX3D((int)seedx+i, (int)seedy+j, (int)seedz+z, bw_ht, bw_wid, bw_len)] = 1;
				count ++;
				//fprintf(stderr, "check point1\n");
				for(int k=0; k<num_clus; k++)
				{
					if(labeled_roi[INDEX3D((int)seedx+i, (int)seedy+j, (int)seedz+z, bw_ht, bw_wid, bw_len)]==k)
					{
						temp_arr[k]++;
						break;
					}
				}// class k
			}// z
		}// j
	}// i
	//fprintf(stderr, "tmp_arr = %d,%d\n", temp_arr[0],temp_arr[1]);
	les_reg_label = argmax_int(temp_arr, num_clus);
	//fprintf(stderr, "label = %d\n", les_reg_label);
	//fprintf(stderr, "count = %d\n", count);
	//fprintf(stderr, "count_temp = %d\n", count_temp);
	while(count_temp<count)
	{
		//fprintf(stderr, "check point1.1\n");
		x_curr = x_ord[count_temp];
		//fprintf(stderr, "x_curr=%d\n", x_curr);
    	y_curr = y_ord[count_temp];
    	z_curr = z_ord[count_temp];
		//fprintf(stderr, "check point1.2\n");
		count_temp ++;
		//fprintf(stderr, "labeled_roi = %d\n",labeled_roi[INDEX3D(x_curr, y_curr, z_curr, bw_ht, bw_wid, bw_len)]);
		if((labeled_roi[INDEX3D(x_curr, y_curr, z_curr, bw_ht, bw_wid, bw_len)]+0.0) == (les_reg_label+0.0))
		{
			//fprintf(stderr, "in\n");
			new_labels[INDEX3D(x_curr, y_curr, z_curr, bw_ht, bw_wid, bw_len)] = 1;
			for(int ia=-1; ia<2; ia++)
			{
				for(int ib=-1; ib<2; ib++)
				{
					for(int ic=-1; ic<2; ic++)
					{
						if(x_curr+ia >=0 && y_curr+ib >= 0 && z_curr+ic >= 0 && x_curr+ia < bw_ht && y_curr+ib < bw_wid && z_curr+ic < bw_len)
						{
							if(flag[INDEX3D(x_curr+ia, y_curr+ib, z_curr+ic, bw_ht, bw_wid, bw_len)] ==0)
							{
								
								flag[INDEX3D(x_curr+ia, y_curr+ib, z_curr+ic, bw_ht, bw_wid, bw_len)] = 1;
								x_ord = append_int_array(x_ord, count, x_curr+ia);
								y_ord = append_int_array(y_ord, count, y_curr+ib);
								z_ord = append_int_array(z_ord, count, z_curr+ic);
								//fprintf(stderr, "z_ord=%d\n", z_ord[count]);
								count++;
								//fprintf(stderr, "check point3\n");
							}
						}
					}//ic
				}//ib
			}//ia
		}
	}
	for(int i=0; i < nvox; ++i)
		labeled_roi[i] = new_labels[i];
	free((void *)flag);
	free((void *)new_labels);
	free_dvector((void *)temp_arr);
	return 0;
}

int MRF_EM_ctinfo(float* voxvals, LABELTYPE* labels, LABELTYPE* CT_label, int xdim, int ydim, int zdim, int num_clus, double beta, double gamma)
{
	LABELTYPE* label_map;
	double* clique_pot;
	double* clique_pot_2;
	double* ini_pot;
	double* prob;
	double gaus_1;
	double* gaus_2;
	double* gaus_3;
	double* gaus_prob;
	double* context_pot;
	double* mean_cluster;
	double* variance_cluster;
	label_map = (LABELTYPE *)malloc(xdim*ydim*zdim*sizeof(LABELTYPE)); // define label map
	gaus_2 = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	gaus_3 = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	gaus_prob    = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	clique_pot   = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	context_pot  = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	clique_pot_2 = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	ini_pot      = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	prob         = (double *)malloc(xdim*ydim*zdim*num_clus*sizeof(double));
	mean_cluster=(double *)dvector(num_clus*sizeof(double)); // class means
	variance_cluster=(double *)dvector(num_clus*sizeof(double)); // class variances

	k_means(voxvals, label_map, xdim, ydim, zdim, num_clus, 0, 1); // k means clustering as initialization
	//for(int i=0; i < xdim*ydim*zdim; ++i)
	//	labels[i] = label_map[i];
	//return 0;

	int nvox = xdim*ydim*zdim; // number of voxels

	for(int k=0; k<num_clus; k++)
	{
		mean_cluster[k]     = mean_class(voxvals, nvox, label_map, k);
		variance_cluster[k] = var_class(voxvals, nvox, label_map, k);
	}//class k
	//fprintf(stderr, "mrf: mean1=%f, mean2=%f,mean3=%f\n", mean_cluster[0], mean_cluster[1], mean_cluster[2]);
	//fprintf(stderr, "mrf: var1=%f, var2=%f,var3=%f\n", variance_cluster[0], variance_cluster[1], variance_cluster[2]);
	//clique_pot = (double *)malloc(xdim*ydim*zdim*sizeof(double));
	//clique_pot_2 = (double *)malloc(xdim*ydim*zdim*sizeof(double));

	
	int max_iter = 20;
	int first_enter_flag = 1;
	LABELTYPE label;
	//fprintf(stderr, "gamma is:%f\n", gamma);
	//fprintf(stderr, "beta is:%f\n", beta);
	for(int iter=0; iter<max_iter; iter++)
	{
		//fprintf(stderr, "iteration=%d\n", iter);
		for(int k=0; k<num_clus; k++)
		{
			if(first_enter_flag == 1)
			{
				for(int ix=0;ix<xdim;ix++)          //x dim
				{
					for(int iy=0;iy<ydim;iy++)      //y dim
					{
						for(int iz=0;iz<zdim;iz++)  //z dim
						{
							double tmp_pot = 0.0;
							for(int ia=-1; ia<2; ia++)
							{
								for(int ib=-1; ib<2; ib++)
								{
									for(int ic=-1; ic<2; ic++)
									{
										if(ix + ia < 0 || iy + ib < 0 || iz + ic < 0 || ix + ia >= xdim || iy + ib >= ydim || iz + ic >= zdim) continue;
										if(ia != 0 || ib != 0 || ic != 0)
										{
											label = label_map[INDEX3D(ix+ia, iy+ib, iz+ic, xdim, ydim, zdim)];
											if(k == label)
											{
												tmp_pot -= 1.0;
											}
											else
											{
												tmp_pot += 1.0;
											}
										}
									}// ic
								}// ib
							}// ia
							clique_pot[INDEX3D(ix, iy, iz, xdim, ydim, zdim)+k*nvox] = tmp_pot;
							//if(INDEX3D(ix, iy, iz, xdim, ydim, zdim)%1000 == 0) fprintf(stderr, "tmp_pot1=%f\n", tmp_pot);
							if(k<num_clus-1)
							{
								context_pot[INDEX3D(ix, iy, iz, xdim, ydim, zdim)+k*nvox] = pow((CT_label[INDEX3D(ix, iy, iz, xdim, ydim, zdim)] - k), 2.0);
							}
							else
							{
								context_pot[INDEX3D(ix, iy, iz, xdim, ydim, zdim)+k*nvox] = 0;
							}
						}// iz
					}// iy
				}// ix
			}// if first enter
			else
			{
				for(int clique_i=0; clique_i<nvox; clique_i++)
				{
					clique_pot[clique_i+k*nvox] = clique_pot_2[clique_i+k*nvox];
				}// clique i
			}
			for(int i=0; i<nvox; i++)
			{
				ini_pot[i+k*nvox] = pow(voxvals[i] - mean_cluster[k],2.0)/variance_cluster[k] + log(variance_cluster[k]);
			}

		}//k class
		//fprintf(stderr, "location 0\n");
		double* tot_pot;
		tot_pot = (double *)dvector(num_clus*sizeof(double));
		for(int i=0; i<nvox; i++)
		{
			for(int k=0; k<num_clus; k++)
			{
				tot_pot[k] = ini_pot[i+k*nvox] + beta*clique_pot[i+k*nvox] + gamma*context_pot[i+k*nvox];
			}
			label_map[i] = argmin_double(tot_pot, num_clus);
		}
		free_dvector((void *)tot_pot);

		double gaus_1;
		////////////////////// EM /////////////////////////////
		for(int k=0; k<num_clus; k++)
		{
			gaus_1 = 1/sqrt(2*3.1415926*sqrt(variance_cluster[k]));
			//fprintf(stderr, "location 2\n");
			for(int i=0; i<nvox; i++)
			{
				gaus_2[i] = -pow((double)voxvals[i]-mean_cluster[k],2.0);
				gaus_3[i] = exp(gaus_2[i]/(2*variance_cluster[k]));
				gaus_prob[i] = gaus_1 * gaus_3[i];
			}
			for(int ix=0;ix<xdim;ix++)          //x dim
			{
				for(int iy=0;iy<ydim;iy++)      //y dim
				{
					for(int iz=0;iz<zdim;iz++)  //z dim
					{
						double tmp_pot = 0.0;
						for(int ia=-1; ia<2; ia++)
						{
							for(int ib=-1; ib<2; ib++)
							{
								for(int ic=-1; ic<2; ic++)
								{

									if(ix + ia < 0 || iy + ib < 0 || iz + ic < 0 || ix + ia >= xdim || iy + ib >= ydim || iz + ic >= zdim) continue;
									if(ia != 0 || ib != 0 || ic != 0)
									{
										label = label_map[INDEX3D(ix+ia, iy+ib, iz+ic, xdim, ydim, zdim)];
										if(k == label)
										{
											tmp_pot -= 1.0;
										}
										else
										{
											tmp_pot += 1.0;
										}
									}
								}// ic
							}// ib
						}// ia
						clique_pot_2[INDEX3D(ix, iy, iz, xdim, ydim, zdim)+k*nvox] = beta*tmp_pot;
					}// iz
				}// iy
			}// ix
			for(int i=0; i<nvox; i++)
			{
				prob[i+k*nvox] = gaus_prob[i]*exp(-clique_pot_2[i+k*nvox]);
				/*
				if(i%20000 == 0) 
				{
					fprintf(stderr, "gaus_prob = %f\n", gaus_prob[i]);
					fprintf(stderr, "-clique_pot_2 = %f\n", -clique_pot_2[i+k*nvox]);
					fprintf(stderr, "exp = %f\n", exp(-clique_pot_2[i+k*nvox]));
				}
				*/
			}

		}// k class
		for(int i=0; i<nvox; i++)
		{
			double tmp_prob=0.0;
			for(int k=0; k<num_clus; k++)
			{
				tmp_prob += prob[i+k*nvox];
			}
			for(int k=0; k<num_clus; k++)
			{
				prob[i+k*nvox] = prob[i+k*nvox]/tmp_prob;
			}
		}

		////////update mean and var//////////
		for(int k=0; k<num_clus; k++)
		{
			double tmp_val_1 = 0.0;
			double tmp_sum   = 0.0;
			double tmp_val_2 = 0.0;
			for(int i=0; i<nvox; i++)
			{
				// for mean
				tmp_val_1 += prob[i+k*nvox]*voxvals[i];
				tmp_sum += prob[i+k*nvox];
			}
			mean_cluster[k] = tmp_val_1/tmp_sum;
			//fprintf(stderr, "tmp_sum=%f\n", tmp_sum);
			for(int i=0; i<nvox; i++)
			{
				// for var
				tmp_val_2 += prob[i+k*nvox]*pow((voxvals[i]-mean_cluster[k]),2.0);
			}
			variance_cluster[k] = tmp_val_2/tmp_sum;
		}
		//fprintf(stderr, "mrf: mean1=%f, mean2=%f,mean3=%f\n", mean_cluster[0], mean_cluster[1], mean_cluster[2]);
		first_enter_flag = 0;

	}//iter
	for(int i=0; i < xdim*ydim*zdim; ++i)
		labels[i] = label_map[i];

	free((void *)context_pot);
	free((void *)clique_pot);
	free((void *)clique_pot_2);
	free((void *)ini_pot);
	free((void *)prob);
	free((void *)gaus_2);
	free((void *)gaus_3);
	free((void *)gaus_prob);
	free((void *)label_map);
	free_dvector((void *)variance_cluster);
	free_dvector((void *)mean_cluster);
	return 0;
}