#include "runtime_api.h"

/// GPisMap (2d)
int create_gpm_instance(GPMHandle *gh){
    *gh = new GPisMap;
    return 1;
}

int delete_gpm_instance(GPMHandle gh){
    if (gh != NULL)
        delete gh;
    return 1;
}


int reset_gpm(GPMHandle gh){
    if (gh != NULL){
        gh->reset();
    }
    return 1;
}

int config_gpm(GPMHandle gh, const char *p_key, void *p_value) {
    if (gh != NULL){
        gh->setParam(p_key, p_value);
        return 1;
    }
    return 0;
}

int update_gpm(GPMHandle gh, float * datax,  float * dataf, int N, float* pose){ // pose[6]
    if (gh != NULL){
        gh->update(datax, dataf, N, pose);
        return 1;
    }
    return 0;
}

int test_gpm(GPMHandle gh, float * x,  int dim,  int leng, float* res){
    if (gh != NULL){
        gh->test(x, dim, leng, res);
        return 1;
    }
    return 0;
}

/// GPisMap3 (3d)
int create_gpm3d_instance(GPM3Handle *gh){
    *gh = new GPisMap3;
    return 1;
}

int delete_gpm3d_instance(GPM3Handle gh){
    if (gh != NULL){
        delete gh;
        gh = NULL;
    }
    return 1;
}

int reset_gpm3d(GPM3Handle gh){
    if (gh != NULL){
        gh->reset();
    }
    return 1;
}

int set_gpm3d_camparam(GPM3Handle gh, 
                       float fx, 
                       float fy,
                       float cx,
                       float cy,
                       int w,
                       int h){
    if (gh != NULL){
        camParam c(fx, fy, cx, cy, w, h);
        gh->resetCam(c);
        return 1;
    }
    return 0;
}

int update_gpm3d(GPM3Handle gh, float * depth, int numel, float* pose){ // pose[12]
    if (gh != NULL){
        gh->update(depth, numel, pose);
        return 1;
    }
    return 0;
}

int test_gpm3d(GPM3Handle gh, float * x,  int dim,  int leng, float* res){
    if (gh != NULL){
        gh->test(x, dim, leng, res);
        return 1;
    }
    return 0;
}