// # -*- mode: c -*-
#pragma once
#define THREADS_PER_BLOCK 512

//=================== ALLOCATE METHODS =============== //

//ints -- these are the same for double and float versions
void gpu_allocate_block_luts(int **p_gpu_cellpair_lut, int **p_gpu_cellthread_lut, const int numblocks);
void gpu_allocate_cell_luts(int **p_gpu_same_cell, int64_t **p_gpu_icell0, int64_t **p_gpu_icell1, const int64_t num_cell_pairs);
void gpu_allocate_lattice_luts(int **p_gpu_np, int **p_gpu_start_idx, const int64_t num_cells);

//doubles
void gpu_allocate_mins_double(double **p_gpu_min_dx, double **p_gpu_min_dy, const int64_t num_cell_pairs);
void gpu_allocate_mocks_double(double **p_X1, double **p_Y1, double **p_Z1, const int64_t ND1);
void gpu_allocate_outputs_double(double **p_gpu_savg, int **p_gpu_npairs, const int totnbins);
void gpu_allocate_one_array_double(double **p_gpu_supp_sqr, const int nsbin);
void gpu_allocate_weight_output_double(double **p_gpu_weightavg, const int totnbins);
void gpu_allocate_weights_double(double **p_weights, const int64_t ND1, uint8_t num_weights);


//floats
void gpu_allocate_mins_float(float **p_gpu_min_dx, float **p_gpu_min_dy, const int64_t num_cell_pairs);
void gpu_allocate_mocks_float(float **p_X1, float **p_Y1, float **p_Z1, const int64_t ND1);
void gpu_allocate_outputs_float(float **p_gpu_savg, int **p_gpu_npairs, const int totnbins);
void gpu_allocate_one_array_float(float **p_gpu_supp_sqr, const int nsbin);
void gpu_allocate_weight_output_float(float **p_gpu_weightavg, const int totnbins);
void gpu_allocate_weights_float(float **p_weights, const int64_t ND1, uint8_t num_weights);



//=================== FREE MEMORY =============== //

//ints -- these are the same for double and float versions
void gpu_free_block_luts(int *gpu_cellpair_lut, int *gpu_cellthread_lut);
void gpu_free_cell_luts(int *gpu_same_cell, int64_t *gpu_icell0, int64_t *gpu_icell1);
void gpu_free_lattice_luts(int *gpu_np, int *gpu_start_idx);


//doubles
void gpu_free_closests_double(double *gpu_closest_x1, double *gpu_closest_y1, double *gpu_closest_z1);
void gpu_free_mins_double(double *gpu_min_dx, double *gpu_min_dy);
void gpu_free_mocks_double(double *X1, double *Y1, double *Z1);
void gpu_free_outputs_double(double *gpu_savg, int *gpu_npairs);
void gpu_free_one_array_double(double *gpu_supp_sqr);
void gpu_free_weight_output_double(double *gpu_weightavg);
void gpu_free_weights_double(double *weights);

//floats
void gpu_free_closests_float(float *gpu_closest_x1, float *gpu_closest_y1, float *gpu_closest_z1);
void gpu_free_mins_float(float *gpu_min_dx, float *gpu_min_dy);
void gpu_free_mocks_float(float *X1, float *Y1, float *Z1);
void gpu_free_outputs_float(float *gpu_savg, int *gpu_npairs);
void gpu_free_one_array_float(float *gpu_supp_sqr);
void gpu_free_weight_output_float(float *gpu_weightavg);
void gpu_free_weights_float(float *weights);


// ================== KERNEL =================== /

int gpu_batch_countpairs_s_mu_mocks_double(double *x0, double *y0, double *z0, double *weights0, uint8_t nw0,
               double *x1, double *y1, double *z1, double *weights1, uint8_t i0,
               const int N, int *np0, int *np1, 
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               double *min_xdiff, double *min_ydiff, 
               double *savg, int *npairs, double *weightavg, const double *supp_sqr,
               const double sqr_smax, const double sqr_smin, const int nsbin,
               const int nmu_bins, 
               const double sqr_mumax, const double inv_dmu, const double mumin_invstep,
               double inv_sstep, double smin_invstep, const selection_struct selection,
               int need_savg, const weight_method_t weight_method, const pair_weight_struct pair_weight,
               double *p_weight, double *p_sep,
               int autocorr, int los_type, int bin_type);


int gpu_batch_countpairs_s_mu_mocks_float(float *x0, float *y0, float *z0, float *weights0, uint8_t nw0,
               float *x1, float *y1, float *z1, float *weights1, uint8_t niw0,
               const int N, int *np0, int *np1,
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               float *min_xdiff, float *min_ydiff,
               float *savg, int *npairs, float *weightavg, const float *supp_sqr,
               const float sqr_smax, const float sqr_smin, const int nsbin,
               const int nmu_bins,
               const float sqr_mumax, const float inv_dmu, const float mumin_invstep,
               float inv_sstep, float smin_invstep, const selection_struct selection,
               int need_savg, const weight_method_t weight_method, const pair_weight_struct pair_weight,
               float *p_weight, float *p_sep,
               int autocorr, int los_type, int bin_type);

// ======================================================= /
// call cudaDeviceSynchronize

void gpu_device_synchronize();

size_t gpu_get_total_mem();

void gpu_print_cuda_error();
