#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <stdint.h>
#include <inttypes.h>

//#include <iostream>
extern "C" {
#include "defs.h"
//#include "function_precision.h"
//#include "utils.h"
//#include "gridlink_utils_double.h"

//#include "weight_functions_double.h"

#include "cellarray_double.h"
#include "cellarray_float.h"

#include "countpairs_s_mu_gpu.h"
#include <cuda_runtime.h>

// Define pair_struct_double here instead of including weight_functions_double
// Info about a particle pair that we will pass to the weight function
typedef struct
{
    double weights0[MAX_NUM_WEIGHTS];
    double weights1[MAX_NUM_WEIGHTS];
    double dx, dy, dz;

    // These will only be present for mock catalogs
    double parx, pary, parz;

    // Add for angular weights
    double costheta;

    double *p_weight;
    double *p_sep;
    int p_num;
    //pair_weight_struct_double pair_weight;

    int num_weights;
    int num_integer_weights;
    int8_t noffset;
    double default_value;
} pair_struct_double;

typedef struct
{
    float weights0[MAX_NUM_WEIGHTS];
    float weights1[MAX_NUM_WEIGHTS];
    float dx, dy, dz;

    // These will only be present for mock catalogs
    float parx, pary, parz;

    // Add for angular weights
    float costheta;

    float *p_weight;
    float *p_sep;
    int p_num;
    //pair_weight_struct_float pair_weight;

    int num_weights;
    int num_integer_weights;
    int8_t noffset;
    float default_value;
} pair_struct_float;

}


//device function to do inverse_bitwise weighting
__device__ double inverse_bitwise_double(pair_struct_double *pair){
    int nbits = pair->noffset;
    for (int w=0;w<pair->num_integer_weights;w++) {
        nbits += __popcll(*((long *) &(pair->weights0[w])) & *((long *) &(pair->weights1[w])));
    }
    double weight = (nbits == 0) ? pair->default_value : 1./nbits;
    int num = pair->p_num;
    if (num) {
        double costheta = pair->costheta;
        double *pair_sep = pair->p_sep;
        if (costheta > pair_sep[num-1] || (costheta <= pair_sep[0])) {
            ;
        }
        else {
            double *pair_weight = pair->p_weight;
            for (int kbin=0;kbin<num-1;kbin++) {
                if(costheta <= pair_sep[kbin+1]) { // ]min, max], as costheta instead of theta
                    double frac = (costheta - pair_sep[kbin])/(pair_sep[kbin+1] - pair_sep[kbin]);
                    weight *= (1 - frac) * pair_weight[kbin] + frac * pair_weight[kbin+1];
                    break;
                }
            }
        }
    }
    num = pair->num_weights;
    int numi = pair->num_integer_weights;
    if (num > numi) weight *= pair->weights0[numi]*pair->weights1[numi]; // multiply by the first float weight
    numi++;
    if (num > numi) weight -= pair->weights0[numi]*pair->weights1[numi]; // subtract the second float weight
    return weight;
}

__device__ float inverse_bitwise_float(pair_struct_float *pair){
    int nbits = pair->noffset;
    for (int w=0;w<pair->num_integer_weights;w++) {
        nbits += __popc(*((int *) &(pair->weights0[w])) & *((int *) &(pair->weights1[w])));
    }
    float weight = (nbits == 0) ? pair->default_value : 1./nbits;
    int num = pair->p_num;
    if (num) {
        float costheta = pair->costheta;
        float *pair_sep = pair->p_sep;
        if (costheta > pair_sep[num-1] || (costheta <= pair_sep[0])) {
            ;
        }
        else {
            float *pair_weight = pair->p_weight;
            for (int kbin=0;kbin<num-1;kbin++) {
                if(costheta <= pair_sep[kbin+1]) { // ]min, max], as costheta instead of theta
                    float frac = (costheta - pair_sep[kbin])/(pair_sep[kbin+1] - pair_sep[kbin]);
                    weight *= (1 - frac) * pair_weight[kbin] + frac * pair_weight[kbin+1];
                    break;
                }
            }
        }
    }
    num = pair->num_weights;
    int numi = pair->num_integer_weights;
    if (num > numi) weight *= pair->weights0[numi]*pair->weights1[numi]; // multiply by the first float weight
    numi++;
    if (num > numi) weight -= pair->weights0[numi]*pair->weights1[numi]; // subtract the second float weight
    return weight;
}

__global__ void countpairs_s_mu_kernel_double(double *x0, double *y0, double *z0,
               double *x1, double *y1, double *z1, int N,
               int *np0, int *np1,
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               double *min_xdiff, double *min_ydiff,
               double *savg, int *npairs, const double *supp_sqr,
               const double pimax, double *off_xwrap, double *off_ywrap, double *off_zwrap,
               const double sqr_smax, const double sqr_smin, const int nsbin,
               const int nmu_bins,
               const double sqr_mumax, const double inv_dmu, const double mumin_invstep,
               double inv_sstep, double smin_invstep, const selection_struct selection,
               int need_savg, int autocorr, int bin_type) {
    //thread index tidx
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= N) return;

    int icellpair = cellpair_lut[blockIdx.x]; //use block index to find cellpair index
    int cell_tidx = cellthread_lut[blockIdx.x] + threadIdx.x; //index within this cellpair from 0 to np0*np1-1

    //icell0, icell1 will translate icellpair to a cell within each lattice
    //start_idx0, start_idx1 then translate to i, j in x0,y0,z0 and x1,y1,z1
    int64_t cellindex0 = icell0[icellpair];
    int64_t cellindex1 = icell1[icellpair];
    //nthreads = np0 * np1 for each cell pair icell
    int this_np0 = np0[cellindex0];
    int this_np1 = np1[cellindex1];
    if (cell_tidx >= this_np0*this_np1) return;

    //start_idx0, start_idx1 give index for first element of x0, y0, z0 and x1, y1, z1 in cell icell.
    //% and / to get i, j
    int i = start_idx0[cellindex0] + cell_tidx / this_np1;
    int j = start_idx1[cellindex1] + cell_tidx % this_np1;

    //get positions for each particle
    double xpos = x0[i] + off_xwrap[icellpair];
    double ypos = y0[i] + off_ywrap[icellpair];
    double zpos = z0[i] + off_zwrap[icellpair];

    double x1pos = x1[j];
    double y1pos = y1[j];
    double z1pos = z1[j];

    if (same_cell[icellpair] && z1pos <= zpos) {
        //return if same particle or in same cell with z1 < z0
        //this way we do not double count pairs
        return;
    }

    double max_dz = sqrt(sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair]);
    max_dz = pimax < max_dz ? pimax:max_dz;
    const double this_dz = z1pos-zpos;
    if (abs(this_dz) >= max_dz) {
        //particle too far away in z
        return;
    }

    const double this_dx = x1pos-xpos;
    const double this_dy = y1pos-ypos;
    const double sqr_min_sep_this_point = this_dx*this_dx + this_dy*this_dy + this_dz*this_dz;
    if (sqr_min_sep_this_point >= sqr_smax || this_dz > pimax) {
        //            if(sqr_min_sep_this_point >= sqr_smax || min_dz > pimax) {
        //particle too far away in separation
        return;
    }

    //const double sqr_max_dz = sqr_smax - min_dx*min_dx - min_dy*min_dy;
    const double sqr_max_dz = sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair];
    max_dz = sqr_max_dz < pimax*pimax ? sqrt(sqr_max_dz):pimax;


    const double dx = x1pos - xpos;
    const double dy = y1pos - ypos;
    const double dz = z1pos - zpos;//the ordering is important. localz1 - zpos ensures dz is in increasing order for future iterations

    //return if greater than max dz
    if (dz > max_dz) {
        return;
    }

    const double sqr_dx_dy = dx*dx + dy*dy;
    //const double sqr_s = perpx*perpx + perpy*perpy + perpz*perpz;
    if((selection.selection_type & RP_SELECTION) && ((sqr_dx_dy < selection.rpmin_sqr) || (sqr_dx_dy >= selection.rpmax_sqr))){
        return;
    }

    const double sqr_dz = dz*dz;
    const double sqr_s = sqr_dx_dy + sqr_dz;
    if(sqr_s >= sqr_smax || sqr_s < sqr_smin) {
        return;
    }

    double sqr_mu = 0.;
    if (sqr_s > 0.) {
        if (sqr_dz >= sqr_s * sqr_mumax) return;
        sqr_mu = sqr_dz/sqr_s;
    }


    double s = 0;
    if(need_savg || bin_type == BIN_LIN) {
        s = sqrt(sqr_s);
    }

    int kbin = 0;
    if (bin_type == BIN_LIN) {
        kbin = (int) (s*inv_sstep + smin_invstep);
    } else {
        for(kbin=nsbin-1;kbin>=1;kbin--) {
            if(sqr_s >= supp_sqr[kbin-1]) {
                break;
            }
        }//finding kbin
    }

    kbin *= nmu_bins+1;
    const double mubin = dz >=0 ? sqrt(sqr_mu)*inv_dmu : -sqrt(sqr_mu)*inv_dmu;
    int ibin = kbin + (int) (mubin + mumin_invstep);
    //use atomic add to guarantee atomicity
    atomicAdd(&npairs[ibin], 1);
    if (need_savg) atomicAdd(&savg[ibin], s);
    if (autocorr == 1) {
        ibin = kbin + (int) (-mubin + mumin_invstep);
        //use atomic add to guarantee atomicity
        atomicAdd(&npairs[ibin], 1);
        if (need_savg) atomicAdd(&savg[ibin], s);
    }
}

__global__ void countpairs_s_mu_kernel_float(float *x0, float *y0, float *z0,
               float *x1, float *y1, float *z1, int N,
               int *np0, int *np1,
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               float *min_xdiff, float *min_ydiff,
               float *savg, int *npairs, const float *supp_sqr,
               const float pimax, float *off_xwrap, float *off_ywrap, float *off_zwrap,
               const float sqr_smax, const float sqr_smin, const int nsbin,
               const int nmu_bins,
               const float sqr_mumax, const float inv_dmu, const float mumin_invstep,
               float inv_sstep, float smin_invstep, const selection_struct selection,
               int need_savg, int autocorr, int bin_type) {
    //thread index tidx
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= N) return;

    int icellpair = cellpair_lut[blockIdx.x]; //use block index to find cellpair index
    int cell_tidx = cellthread_lut[blockIdx.x] + threadIdx.x; //index within this cellpair from 0 to np0*np1-1

    //icell0, icell1 will translate icellpair to a cell within each lattice
    //start_idx0, start_idx1 then translate to i, j in x0,y0,z0 and x1,y1,z1
    int64_t cellindex0 = icell0[icellpair];
    int64_t cellindex1 = icell1[icellpair];
    //nthreads = np0 * np1 for each cell pair icell
    int this_np0 = np0[cellindex0];
    int this_np1 = np1[cellindex1];
    if (cell_tidx >= this_np0*this_np1) return;

    //start_idx0, start_idx1 give index for first element of x0, y0, z0 and x1, y1, z1 in cell icell.
    //% and / to get i, j
    int i = start_idx0[cellindex0] + cell_tidx / this_np1;
    int j = start_idx1[cellindex1] + cell_tidx % this_np1;

    //get positions for each particle
    float xpos = x0[i] + off_xwrap[icellpair];
    float ypos = y0[i] + off_ywrap[icellpair];
    float zpos = z0[i] + off_zwrap[icellpair];

    float x1pos = x1[j];
    float y1pos = y1[j];
    float z1pos = z1[j];

    if (same_cell[icellpair] && z1pos <= zpos) {
        //return if same particle or in same cell with z1 < z0
        //this way we do not double count pairs
        if (z1pos < zpos) return;
        if (z1pos == zpos && j <= i) return;
        //return;
    }

    float max_dz = sqrt(sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair]);
    max_dz = pimax < max_dz ? pimax:max_dz;
    const float this_dz = z1pos-zpos;
    if (abs(this_dz) >= max_dz) {
        //particle too far away in z
        return;
    }

    const float this_dx = x1pos-xpos;
    const float this_dy = y1pos-ypos;
    const float sqr_min_sep_this_point = this_dx*this_dx + this_dy*this_dy + this_dz*this_dz;
    if (sqr_min_sep_this_point >= sqr_smax || this_dz > pimax) {
        //            if(sqr_min_sep_this_point >= sqr_smax || min_dz > pimax) {
        //particle too far away in separation
        return;
    }

    //const float sqr_max_dz = sqr_smax - min_dx*min_dx - min_dy*min_dy;
    const float sqr_max_dz = sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair];
    max_dz = sqr_max_dz < pimax*pimax ? sqrt(sqr_max_dz):pimax;


    const float dx = x1pos - xpos;
    const float dy = y1pos - ypos;
    const float dz = z1pos - zpos;//the ordering is important. localz1 - zpos ensures dz is in increasing order for future iterations

    //return if greater than max dz
    if (dz > max_dz) {
        return;
    }

    const float sqr_dx_dy = dx*dx + dy*dy;
    //const float sqr_s = perpx*perpx + perpy*perpy + perpz*perpz;
    if((selection.selection_type & RP_SELECTION) && ((sqr_dx_dy < selection.rpmin_sqr) || (sqr_dx_dy >= selection.rpmax_sqr))){
        return;
    }

    const float sqr_dz = dz*dz;
    const float sqr_s = sqr_dx_dy + sqr_dz;
    if(sqr_s >= sqr_smax || sqr_s < sqr_smin) {
        return;
    }

    float sqr_mu = 0.;
    if (sqr_s > 0.) {
        if (sqr_dz >= sqr_s * sqr_mumax) return;
        sqr_mu = sqr_dz/sqr_s;
    }


    float s = 0;
    if(need_savg || bin_type == BIN_LIN) {
        s = sqrt(sqr_s);
    }

    int kbin = 0;
    if (bin_type == BIN_LIN) {
        kbin = (int) (s*inv_sstep + smin_invstep);
    } else {
        for(kbin=nsbin-1;kbin>=1;kbin--) {
            if(sqr_s >= supp_sqr[kbin-1]) {
                break;
            }
        }//finding kbin
    }

    kbin *= nmu_bins+1;
    const float mubin = dz >=0 ? sqrt(sqr_mu)*inv_dmu : -sqrt(sqr_mu)*inv_dmu;
    int ibin = kbin + (int) (mubin + mumin_invstep);
    //use atomic add to guarantee atomicity
    atomicAdd(&npairs[ibin], 1);
    if (need_savg) atomicAdd(&savg[ibin], s);
    if (autocorr == 1) {
        ibin = kbin + (int) (-mubin + mumin_invstep);
        //use atomic add to guarantee atomicity
        atomicAdd(&npairs[ibin], 1);
        if (need_savg) atomicAdd(&savg[ibin], s);
    }
}

__global__ void countpairs_s_mu_pair_weights_kernel_double(double *x0, double *y0, double *z0,
               double *weights0, int numweights,
               double *x1, double *y1, double *z1,
               double *weights1, int numintweights,
               int N, int *np0, int *np1,
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               double *min_xdiff, double *min_ydiff,
               double *savg, int *npairs, double *weightavg, const double *supp_sqr,
               const double pimax, double *off_xwrap, double *off_ywrap, double *off_zwrap,
               const double sqr_smax, const double sqr_smin, const int nsbin,
               const int nmu_bins,
               const double sqr_mumax, const double inv_dmu, const double mumin_invstep,
               double inv_sstep, double smin_invstep, const selection_struct selection,
               int need_savg, int need_weightavg, int autocorr, int los_type, int bin_type,
               const weight_method_t weight_method, const pair_weight_struct pair_w, double *p_weight, double *p_sep) {
    //thread index tidx
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= N) return;

    int icellpair = cellpair_lut[blockIdx.x]; //use block index to find cellpair index
    int cell_tidx = cellthread_lut[blockIdx.x] + threadIdx.x; //index within this cellpair from 0 to np0*np1-1

    //icell0, icell1 will translate icellpair to a cell within each lattice
    //start_idx0, start_idx1 then translate to i, j in x0,y0,z0 and x1,y1,z1
    int64_t cellindex0 = icell0[icellpair];
    int64_t cellindex1 = icell1[icellpair];
    //nthreads = np0 * np1 for each cell pair icell
    int this_np0 = np0[cellindex0];
    int this_np1 = np1[cellindex1];
    if (cell_tidx >= this_np0*this_np1) return;

    //start_idx0, start_idx1 give index for first element of x0, y0, z0 and x1, y1, z1 in cell icell.
    //% and / to get i, j
    int i = start_idx0[cellindex0] + cell_tidx / this_np1;
    int j = start_idx1[cellindex1] + cell_tidx % this_np1;

    //get positions for each particle
    double xpos = x0[i] + off_xwrap[icellpair];
    double ypos = y0[i] + off_ywrap[icellpair];
    double zpos = z0[i] + off_zwrap[icellpair];

    double x1pos = x1[j];
    double y1pos = y1[j];
    double z1pos = z1[j];

    if (same_cell[icellpair] && z1pos <= zpos) {
        //return if same particle or in same cell with z1 < z0
        //this way we do not double count pairs
        return;
    }

    double max_dz = sqrt(sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair]);
    max_dz = pimax < max_dz ? pimax:max_dz;
    const double this_dz = z1pos-zpos;
    if (abs(this_dz) >= max_dz) {
        //particle too far away in z
        return;
    }

    const double this_dx = x1pos-xpos;
    const double this_dy = y1pos-ypos;
    const double sqr_min_sep_this_point = this_dx*this_dx + this_dy*this_dy + this_dz*this_dz;
    if (sqr_min_sep_this_point >= sqr_smax || this_dz > pimax) {
        //            if(sqr_min_sep_this_point >= sqr_smax || min_dz > pimax) {
        //particle too far away in separation
        return;
    }

    //const double sqr_max_dz = sqr_smax - min_dx*min_dx - min_dy*min_dy;
    const double sqr_max_dz = sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair];
    max_dz = sqr_max_dz < pimax*pimax ? sqrt(sqr_max_dz):pimax;

    //norm calcs are done if need_costheta == need_weightavg
    //need_weightavg is true by definition in this kernel so remove conditional
    //positions are not divided by norm here
    const double norm1 = sqrt(x1pos*x1pos + y1pos*y1pos + z1pos*z1pos);

    //need_weightavg is true by definition in this kernel so remove conditional
    const double norm0 = sqrt(xpos*xpos + ypos*ypos + zpos*zpos);

    const double dx = x1pos - xpos;
    const double dy = y1pos - ypos;
    const double dz = z1pos - zpos;//the ordering is important. localz1 - zpos ensures dz is in increasing order for future iterations

    //return if greater than max dz
    if (dz > max_dz) {
        return;
    }
    double pair_costheta_d = x1pos*xpos + y1pos*ypos + z1pos*zpos;
    pair_costheta_d /= norm1*norm0;
    if((selection.selection_type & THETA_SELECTION) && ((pair_costheta_d <= selection.costhetamin) || (pair_costheta_d > selection.costhetamax))) return;

    const double sqr_dx_dy = dx*dx + dy*dy;
    //const double sqr_s = perpx*perpx + perpy*perpy + perpz*perpz;
    if((selection.selection_type & RP_SELECTION) && ((sqr_dx_dy < selection.rpmin_sqr) || (sqr_dx_dy >= selection.rpmax_sqr))){
        return;
    }

    const double sqr_dz = dz*dz;
    const double sqr_s = sqr_dx_dy + sqr_dz;
    if(sqr_s >= sqr_smax || sqr_s < sqr_smin) {
        return;
    }

    double sqr_mu = 0.;
    if (sqr_s > 0.) {
        if (sqr_dz >= sqr_s * sqr_mumax) return;
        sqr_mu = sqr_dz/sqr_s;
    }

    double s = 0, pairweight = 0;
    if(need_savg || bin_type == BIN_LIN) {
        s = sqrt(sqr_s);
    }
    if (autocorr) weights1 = weights0;

    //need_weightavg is TRUE so remove conditional and always calculate
    //pairweight - only do simple PAIR_PRODUCT in this kernel
    if (weight_method == PAIR_PRODUCT) pairweight = weights0[i*numweights] * weights1[j*numweights];
    else if (weight_method == INVERSE_BITWISE) {
        //use pair_struct and helper method to calculate inverse bitwise weights
        pair_struct_double pair = {.num_weights=numweights, .num_integer_weights=numintweights};
        for(int w = 0; w < pair.num_weights; w++) {
            pair.weights0[w] = weights0[i*numweights+w];
            pair.weights1[w] = weights1[j*numweights+w];
        }

        pair.dx = dx;
        pair.dy = dy;
        pair.dz = dz;
        pair.costheta = pair_costheta_d;

        pair.p_weight = p_weight;
        pair.p_sep = p_sep;
        pair.p_num = (int)pair_w.num;
        pair.noffset = pair_w.noffset;
        pair.default_value = (double) pair_w.default_value;

        pairweight = inverse_bitwise_double(&pair);
    }


    int kbin = 0;
    if (bin_type == BIN_LIN) {
        kbin = (int) (s*inv_sstep + smin_invstep);
    } else {
        for(kbin=nsbin-1;kbin>=1;kbin--) {
            if(sqr_s >= supp_sqr[kbin-1]) {
                break;
            }
        }//finding kbin
    }

    kbin *= nmu_bins+1;
    const double mubin = dz >=0 ? sqrt(sqr_mu)*inv_dmu : -sqrt(sqr_mu)*inv_dmu;
    int ibin = kbin + (int) (mubin + mumin_invstep);
    //use atomic add to guarantee atomicity
    atomicAdd(&npairs[ibin], 1);
    if (need_savg) {
        //need_weightavg is true so remove conditional
        s*=pairweight;
        atomicAdd(&savg[ibin], s);
    }
    //need_weightavg is true so remove conditional
    atomicAdd(&weightavg[ibin], pairweight); //need_weightavg is always true

    if (autocorr == 1) {
        ibin = kbin + (int) (-mubin + mumin_invstep);
        //use atomic add to guarantee atomicity
        atomicAdd(&npairs[ibin], 1);
        if (need_savg) atomicAdd(&savg[ibin], s); //already weighted
        //need_weightavg is true so remove conditional
        atomicAdd(&weightavg[ibin], pairweight); //need_weightavg is always true
    }
}

__global__ void countpairs_s_mu_pair_weights_kernel_float(float *x0, float *y0, float *z0,
               float *weights0, int numweights,
               float *x1, float *y1, float *z1,
               float *weights1, int numintweights,
               int N, int *np0, int *np1,
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               float *min_xdiff, float *min_ydiff,
               float *savg, int *npairs, float *weightavg, const float *supp_sqr,
               const float pimax, float *off_xwrap, float *off_ywrap, float *off_zwrap,
               const float sqr_smax, const float sqr_smin, const int nsbin,
               const int nmu_bins,
               const float sqr_mumax, const float inv_dmu, const float mumin_invstep,
               float inv_sstep, float smin_invstep, const selection_struct selection,
               int need_savg, int need_weightavg, int autocorr, int los_type, int bin_type,
               const weight_method_t weight_method, const pair_weight_struct pair_w, float *p_weight, float *p_sep) {
    //thread index tidx
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= N) return;

    int icellpair = cellpair_lut[blockIdx.x]; //use block index to find cellpair index
    int cell_tidx = cellthread_lut[blockIdx.x] + threadIdx.x; //index within this cellpair from 0 to np0*np1-1

    //icell0, icell1 will translate icellpair to a cell within each lattice
    //start_idx0, start_idx1 then translate to i, j in x0,y0,z0 and x1,y1,z1
    int64_t cellindex0 = icell0[icellpair];
    int64_t cellindex1 = icell1[icellpair];
    //nthreads = np0 * np1 for each cell pair icell
    int this_np0 = np0[cellindex0];
    int this_np1 = np1[cellindex1];
    if (cell_tidx >= this_np0*this_np1) return;

    //start_idx0, start_idx1 give index for first element of x0, y0, z0 and x1, y1, z1 in cell icell.
    //% and / to get i, j
    int i = start_idx0[cellindex0] + cell_tidx / this_np1;
    int j = start_idx1[cellindex1] + cell_tidx % this_np1;

    //get positions for each particle
    float xpos = x0[i] + off_xwrap[icellpair];
    float ypos = y0[i] + off_ywrap[icellpair];
    float zpos = z0[i] + off_zwrap[icellpair];

    float x1pos = x1[j];
    float y1pos = y1[j];
    float z1pos = z1[j];

    if (same_cell[icellpair] && z1pos <= zpos) {
        //return if same particle or in same cell with z1 < z0
        //this way we do not float count pairs
        if (z1pos < zpos) return;
        if (z1pos == zpos && j <= i) return;
        //return;
    }

    float max_dz = sqrt(sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair]);
    max_dz = pimax < max_dz ? pimax:max_dz;
    const float this_dz = z1pos-zpos;
    if (abs(this_dz) >= max_dz) {
        //particle too far away in z
        return;
    }

    const float this_dx = x1pos-xpos;
    const float this_dy = y1pos-ypos;
    const float sqr_min_sep_this_point = this_dx*this_dx + this_dy*this_dy + this_dz*this_dz;
    if (sqr_min_sep_this_point >= sqr_smax || this_dz > pimax) {
        //            if(sqr_min_sep_this_point >= sqr_smax || min_dz > pimax) {
        //particle too far away in separation
        return;
    }

    //const float sqr_max_dz = sqr_smax - min_dx*min_dx - min_dy*min_dy;
    const float sqr_max_dz = sqr_smax - min_xdiff[icellpair]*min_xdiff[icellpair] - min_ydiff[icellpair]*min_ydiff[icellpair];
    max_dz = sqr_max_dz < pimax*pimax ? sqrt(sqr_max_dz):pimax;

    //norm calcs are done if need_costheta == need_weightavg
    //need_weightavg is true by definition in this kernel so remove conditional
    //positions are not divided by norm here
    const float norm1 = sqrt(x1pos*x1pos + y1pos*y1pos + z1pos*z1pos);

    //need_weightavg is true by definition in this kernel so remove conditional
    const float norm0 = sqrt(xpos*xpos + ypos*ypos + zpos*zpos);

    const float dx = x1pos - xpos;
    const float dy = y1pos - ypos;
    const float dz = z1pos - zpos;//the ordering is important. localz1 - zpos ensures dz is in increasing order for future iterations

    //return if greater than max dz
    if (dz > max_dz) {
        return;
    }

    const float sqr_dx_dy = dx*dx + dy*dy;
    //const float sqr_s = perpx*perpx + perpy*perpy + perpz*perpz;
    if((selection.selection_type & RP_SELECTION) && ((sqr_dx_dy < selection.rpmin_sqr) || (sqr_dx_dy >= selection.rpmax_sqr))){
        return;
    }

    const float sqr_dz = dz*dz;
    const float sqr_s = sqr_dx_dy + sqr_dz;
    if(sqr_s >= sqr_smax || sqr_s < sqr_smin) {
        return;
    }

    float sqr_mu = 0.;
    if (sqr_s > 0.) {
        if (sqr_dz >= sqr_s * sqr_mumax) return;
        sqr_mu = sqr_dz/sqr_s;
    }

    float s = 0, pairweight = 0;
    if(need_savg || bin_type == BIN_LIN) {
        s = sqrt(sqr_s);
    }
    if (autocorr) weights1 = weights0;

    //need_weightavg is TRUE so remove conditional and always calculate
    //pairweight - only do simple PAIR_PRODUCT in this kernel
    if (weight_method == PAIR_PRODUCT) pairweight = weights0[i*numweights] * weights1[j*numweights];
    else if (weight_method == INVERSE_BITWISE) {
        //use pair_struct and helper method to calculate inverse bitwise weights
        pair_struct_float pair = {.num_weights=numweights, .num_integer_weights=numintweights};
        for(int w = 0; w < pair.num_weights; w++) {
            pair.weights0[w] = weights0[i*numweights+w];
            pair.weights1[w] = weights1[j*numweights+w];
        }
        float pair_costheta_d = x1pos*xpos + y1pos*ypos + z1pos*zpos;
        pair_costheta_d /= norm1*norm0;

        pair.dx = dx;
        pair.dy = dy;
        pair.dz = dz;
        pair.costheta = pair_costheta_d;

        pair.p_weight = p_weight;
        pair.p_sep = p_sep;
        pair.p_num = (int)pair_w.num;
        pair.noffset = pair_w.noffset;
        pair.default_value = (float) pair_w.default_value;

        pairweight = inverse_bitwise_float(&pair);
    }

    int kbin = 0;
    if (bin_type == BIN_LIN) {
        kbin = (int) (s*inv_sstep + smin_invstep);
    } else {
        for(kbin=nsbin-1;kbin>=1;kbin--) {
            if(sqr_s >= supp_sqr[kbin-1]) {
                break;
            }
        }//finding kbin
    }

    kbin *= nmu_bins+1;
    const float mubin = dz >=0 ? sqrt(sqr_mu)*inv_dmu : -sqrt(sqr_mu)*inv_dmu;
    int ibin = kbin + (int) (mubin + mumin_invstep);
    //use atomic add to guarantee atomicity
    atomicAdd(&npairs[ibin], 1);
    if (need_savg) {
        //need_weightavg is true so remove conditional
        s*=pairweight;
        atomicAdd(&savg[ibin], s);
    }
    //need_weightavg is true so remove conditional
    atomicAdd(&weightavg[ibin], pairweight); //need_weightavg is always true

    if (autocorr == 1) {
        ibin = kbin + (int) (-mubin + mumin_invstep);
        //use atomic add to guarantee atomicity
        atomicAdd(&npairs[ibin], 1);
        if (need_savg) atomicAdd(&savg[ibin], s); //already weighted
        //need_weightavg is true so remove conditional
        atomicAdd(&weightavg[ibin], pairweight); //need_weightavg is always true
    }
}

extern "C" {

//=================== ALLOCATE METHODS =============== //

// ---------- ints ----------

void gpu_allocate_block_luts(int **p_gpu_cellpair_lut, int **p_gpu_cellthread_lut, const int numblocks) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_cellpair_lut), numblocks*sizeof(int));
    cudaMallocManaged(&(*p_gpu_cellthread_lut), numblocks*sizeof(int));
}

void gpu_allocate_cell_luts(int **p_gpu_same_cell, int64_t **p_gpu_icell0, int64_t **p_gpu_icell1, const int64_t num_cell_pairs) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_same_cell), num_cell_pairs*sizeof(int));
    cudaMallocManaged(&(*p_gpu_icell0), num_cell_pairs*sizeof(int64_t));
    cudaMallocManaged(&(*p_gpu_icell1), num_cell_pairs*sizeof(int64_t));
}

void gpu_allocate_lattice_luts(int **p_gpu_np, int **p_gpu_start_idx, const int64_t num_cells) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_np), num_cells*sizeof(int));
    cudaMallocManaged(&(*p_gpu_start_idx), num_cells*sizeof(int));
}


// ----------- doubles --------------

void gpu_allocate_mins_double(double **p_gpu_min_dx, double **p_gpu_min_dy, const int64_t num_cell_pairs) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_min_dx), num_cell_pairs*sizeof(double));
    cudaMallocManaged(&(*p_gpu_min_dy), num_cell_pairs*sizeof(double));
}

void gpu_allocate_wraps_double(double **p_gpu_xwrap, double **p_gpu_ywrap, double **p_gpu_zwrap, const int64_t num_cell_pairs) {
    cudaMallocManaged(&(*p_gpu_xwrap), num_cell_pairs*sizeof(double));
    cudaMallocManaged(&(*p_gpu_ywrap), num_cell_pairs*sizeof(double));
    cudaMallocManaged(&(*p_gpu_zwrap), num_cell_pairs*sizeof(double));
}

void gpu_allocate_cellarray_double(double **p_X1, double **p_Y1, double **p_Z1, const int64_t ND1) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_X1), ND1*sizeof(double));
    cudaMallocManaged(&(*p_Y1), ND1*sizeof(double));
    cudaMallocManaged(&(*p_Z1), ND1*sizeof(double));
}

void gpu_allocate_outputs_double(double **p_gpu_savg, int **p_gpu_npairs, const int totnbins) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_savg), totnbins*sizeof(double));
    cudaMallocManaged(&(*p_gpu_npairs), totnbins*sizeof(int));
}

void gpu_allocate_one_array_double(double **p_gpu_supp_sqr, const int nsbin) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_supp_sqr), nsbin*sizeof(double));
}

void gpu_allocate_weight_output_double(double **p_gpu_weightavg, const int totnbins) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_weightavg), totnbins*sizeof(double));
}

void gpu_allocate_weights_double(double **p_weights, const int64_t ND1, uint8_t num_weights) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_weights), ND1*num_weights*sizeof(double));
}

// --------------- floats --------------- //

void gpu_allocate_mins_float(float **p_gpu_min_dx, float **p_gpu_min_dy, const int64_t num_cell_pairs) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_min_dx), num_cell_pairs*sizeof(float));
    cudaMallocManaged(&(*p_gpu_min_dy), num_cell_pairs*sizeof(float));
}

void gpu_allocate_wraps_float(float **p_gpu_xwrap, float **p_gpu_ywrap, float **p_gpu_zwrap, const int64_t num_cell_pairs) {
    cudaMallocManaged(&(*p_gpu_xwrap), num_cell_pairs*sizeof(float));
    cudaMallocManaged(&(*p_gpu_ywrap), num_cell_pairs*sizeof(float));
    cudaMallocManaged(&(*p_gpu_zwrap), num_cell_pairs*sizeof(float));
}

void gpu_allocate_cellarray_float(float **p_X1, float **p_Y1, float **p_Z1, const int64_t ND1) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_X1), ND1*sizeof(float));
    cudaMallocManaged(&(*p_Y1), ND1*sizeof(float));
    cudaMallocManaged(&(*p_Z1), ND1*sizeof(float));
}

void gpu_allocate_outputs_float(float **p_gpu_savg, int **p_gpu_npairs, const int totnbins) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_savg), totnbins*sizeof(float));
    cudaMallocManaged(&(*p_gpu_npairs), totnbins*sizeof(int));
}

void gpu_allocate_one_array_float(float **p_gpu_supp_sqr, const int nsbin) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_supp_sqr), nsbin*sizeof(float));
}

void gpu_allocate_weight_output_float(float **p_gpu_weightavg, const int totnbins) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_gpu_weightavg), totnbins*sizeof(float));
}

void gpu_allocate_weights_float(float **p_weights, const int64_t ND1, uint8_t num_weights) {
    // Allocate Unified Memory – accessible from CPU or GPU
    // Takes pointers as args
    cudaMallocManaged(&(*p_weights), ND1*num_weights*sizeof(float));
}

// ============  FREE MEMORY ============= //

// ---------- ints ----------
void gpu_free_block_luts(int *gpu_cellpair_lut, int *gpu_cellthread_lut) {
    cudaFree(gpu_cellpair_lut);
    cudaFree(gpu_cellthread_lut);
}

void gpu_free_cell_luts(int *gpu_same_cell, int64_t *gpu_icell0, int64_t *gpu_icell1) {
    cudaFree(gpu_same_cell);
    cudaFree(gpu_icell0);
    cudaFree(gpu_icell1);
}

void gpu_free_lattice_luts(int *gpu_np, int *gpu_start_idx) {
    cudaFree(gpu_np);
    cudaFree(gpu_start_idx);
}

// ----------- doubles --------------

void gpu_free_mins_double(double *gpu_min_dx, double *gpu_min_dy) {
    cudaFree(gpu_min_dx);
    cudaFree(gpu_min_dy);
}

void gpu_free_wraps_double(double *gpu_xwrap, double *gpu_ywrap, double *gpu_zwrap) {
    cudaFree(gpu_xwrap);
    cudaFree(gpu_ywrap);
    cudaFree(gpu_zwrap);
}

void gpu_free_cellarray_double(double *X1, double *Y1, double *Z1) {
    cudaFree(X1);
    cudaFree(Y1);
    cudaFree(Z1);
}

void gpu_free_outputs_double(double *gpu_savg, int *gpu_npairs) {
    cudaFree(gpu_savg);
    cudaFree(gpu_npairs);
}

void gpu_free_one_array_double(double *gpu_supp_sqr) {
    cudaFree(gpu_supp_sqr);
}

void gpu_free_weight_output_double(double *gpu_weightavg) {
    cudaFree(gpu_weightavg);
}

void gpu_free_weights_double(double *weights) {
    cudaFree(weights);
}

// --------------- floats --------------- //

void gpu_free_mins_float(float *gpu_min_dx, float *gpu_min_dy) {
    cudaFree(gpu_min_dx);
    cudaFree(gpu_min_dy);
}

void gpu_free_wraps_float(float *gpu_xwrap, float *gpu_ywrap, float *gpu_zwrap) {
    cudaFree(gpu_xwrap);
    cudaFree(gpu_ywrap);
    cudaFree(gpu_zwrap);
}

void gpu_free_cellarray_float(float *X1, float *Y1, float *Z1) {
    cudaFree(X1);
    cudaFree(Y1);
    cudaFree(Z1);
}

void gpu_free_outputs_float(float *gpu_savg, int *gpu_npairs) {
    cudaFree(gpu_savg);
    cudaFree(gpu_npairs);
}

void gpu_free_one_array_float(float *gpu_supp_sqr) {
    cudaFree(gpu_supp_sqr);
}

void gpu_free_weight_output_float(float *gpu_weightavg) {
    cudaFree(gpu_weightavg);
}

void gpu_free_weights_float(float *weights) {
    cudaFree(weights);
}

//==========================//


void gpu_device_synchronize() {
  // Wait for GPU to finish before accessing on host
  //This does not need to be called after every kernel invocation,
  //but just before memory is accessed on host
  cudaDeviceSynchronize();
}

// =========   Kernel called below ============//

int gpu_batch_countpairs_s_mu_double(double *x0, double *y0, double *z0,
               double *weights0, uint8_t numweights,
               double *x1, double *y1, double *z1,
               double *weights1, uint8_t numintweights,
               const int N, int *np0, int *np1,
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               double *min_xdiff, double *min_ydiff,
               double *savg, int *npairs, double *weightavg, const double *supp_sqr,
               const double pimax, double *off_xwrap, double *off_ywrap, double *off_zwrap,
               const double sqr_smax, const double sqr_smin, const int nsbin,
               const int nmu_bins,
               const double sqr_mumax, const double inv_dmu, const double mumin_invstep,
               double inv_sstep, double smin_invstep, const selection_struct selection,
               int need_savg, const weight_method_t weight_method, const pair_weight_struct pair_weight,
               double *p_weight, double *p_sep,
               int autocorr, int los_type, int bin_type) {
    long threads = N;
    int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

    //select kernel based on weight_method - faster to have a base kernel that
    //is not unnecessarily passed extra arrays for weighting calcs that won't
    //be performed

    if (weight_method == NONE) {
        countpairs_s_mu_kernel_double<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            x0, y0, z0,
            x1, y1, z1, N,
            np0, np1,
            same_cell, icell0, icell1,
            cellpair_lut, cellthread_lut,
            start_idx0, start_idx1,
            min_xdiff, min_ydiff,
            savg, npairs, supp_sqr,
            pimax, off_xwrap, off_ywrap, off_zwrap,
            sqr_smax, sqr_smin, nsbin, nmu_bins,
            sqr_mumax,inv_dmu,mumin_invstep,
            inv_sstep, smin_invstep, selection,
            need_savg, autocorr, bin_type);
    } else {
        countpairs_s_mu_pair_weights_kernel_double<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            x0, y0, z0, weights0, (int)numweights,
            x1, y1, z1, weights1, (int)numintweights,
            N, np0, np1,
            same_cell, icell0, icell1,
            cellpair_lut, cellthread_lut,
            start_idx0, start_idx1,
            min_xdiff, min_ydiff,
            savg, npairs, weightavg, supp_sqr,
            pimax, off_xwrap, off_ywrap, off_zwrap,
            sqr_smax, sqr_smin, nsbin, nmu_bins,
            sqr_mumax,inv_dmu,mumin_invstep,
            inv_sstep, smin_invstep, selection,
            need_savg, 1, autocorr, los_type, bin_type,
            weight_method, pair_weight, p_weight, p_sep);
    }

    //synchronize memory after kernel call
    cudaDeviceSynchronize();
//    gpu_print_cuda_error();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

// ----------- float version ----------------

int gpu_batch_countpairs_s_mu_float(float *x0, float *y0, float *z0,
               float *weights0, uint8_t numweights,
               float *x1, float *y1, float *z1,
               float *weights1, uint8_t numintweights,
               const int N, int *np0, int *np1,
               int *same_cell, int64_t *icell0, int64_t *icell1,
               int *cellpair_lut, int *cellthread_lut,
               int *start_idx0, int *start_idx1,
               float *min_xdiff, float *min_ydiff,
               float *savg, int *npairs, float *weightavg, const float *supp_sqr,
               const float pimax, float *off_xwrap, float *off_ywrap, float *off_zwrap,
               const float sqr_smax, const float sqr_smin, const int nsbin,
               const int nmu_bins,
               const float sqr_mumax, const float inv_dmu, const float mumin_invstep,
               float inv_sstep, float smin_invstep, const selection_struct selection,
               int need_savg, const weight_method_t weight_method, const pair_weight_struct pair_weight,
               float *p_weight, float *p_sep,
               int autocorr, int los_type, int bin_type) {

    long threads = N;
    int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

    //select kernel based on weight_method - faster to have a base kernel that
    //is not unnecessarily passed extra arrays for weighting calcs that won't
    //be performed

    if (weight_method == NONE) {
        countpairs_s_mu_kernel_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            x0, y0, z0,
            x1, y1, z1, N,
            np0, np1,
            same_cell, icell0, icell1,
            cellpair_lut, cellthread_lut,
            start_idx0, start_idx1,
            min_xdiff, min_ydiff,
            savg, npairs, supp_sqr,
            pimax, off_xwrap, off_ywrap, off_zwrap,
            sqr_smax, sqr_smin, nsbin, nmu_bins,
            sqr_mumax,inv_dmu,mumin_invstep,
            inv_sstep, smin_invstep, selection,
            need_savg, autocorr, bin_type);
    } else {
        countpairs_s_mu_pair_weights_kernel_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            x0, y0, z0, weights0, (int)numweights,
            x1, y1, z1, weights1, (int)numintweights,
            N, np0, np1,
            same_cell, icell0, icell1,
            cellpair_lut, cellthread_lut,
            start_idx0, start_idx1,
            min_xdiff, min_ydiff,
            savg, npairs, weightavg, supp_sqr,
            pimax, off_xwrap, off_ywrap, off_zwrap,
            sqr_smax, sqr_smin, nsbin, nmu_bins,
            sqr_mumax,inv_dmu,mumin_invstep,
            inv_sstep, smin_invstep, selection,
            need_savg, 1, autocorr, los_type, bin_type,
            weight_method, pair_weight, p_weight, p_sep);
    }

    //synchronize memory after kernel call
    cudaDeviceSynchronize();
//    gpu_print_cuda_error();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

size_t gpu_get_total_mem() {
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }
    return total_byte;
}

void gpu_print_cuda_error() {
       size_t free_byte ;

        size_t total_byte ;

        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){

            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

            exit(1);

        }

        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

        cudaError_t err = cudaGetLastError();
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
}

//==============================================
}
