/*
 * GPisMap - Online Continuous Mapping using Gaussian Process Implicit Surfaces
 * https://github.com/leebhoram/GPisMap
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License v3 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of any FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License v3 for more details.
 *
 * You should have received a copy of the GNU General Public License v3
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-3.0.html.
 *
 * Authors: Bhoram Lee <bhoram.lee@gmail.com>
 */
#include "GPisMap3.h"
#include <chrono>
#include <thread>
//
static FLOAT Rtimes = (FLOAT)2.0;
static FLOAT C_leng = (FLOAT)0.025;
tree_param OcTree::param = tree_param((FLOAT)(0.0125/2.0),(FLOAT)1.6,(FLOAT)0.4, C_leng);
// Note: 1.6 = 0.0125*(2.0^7)
//       0.4 = 0.0125*(2.0^5)

static std::chrono::high_resolution_clock::time_point t1;
static std::chrono::high_resolution_clock::time_point t2;

#define MAX_RANGE   4e0
#define MIN_RANGE   4e-1

static inline bool isRangeValid(FLOAT r)
{
    return (r < MAX_RANGE) &&  (r > MIN_RANGE);
}

static inline FLOAT occ_test(FLOAT rinv, FLOAT rinv0, FLOAT a)
{
    return 2.0*(1.0/(1.0+exp(-a*(rinv-rinv0)))-0.5);
}

static inline FLOAT saturate(FLOAT val, FLOAT min_val, FLOAT max_val)
{
    return std::min(std::max(val,min_val),max_val);
}

static std::array<FLOAT,9> quat2dcm(FLOAT q[4]){
    std::array<FLOAT,9> dcm;
    dcm[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    dcm[1] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    dcm[2] = 2.0*(q[1]*q[3] - q[0]*q[2]);

    dcm[3] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    dcm[4] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    dcm[5] = 2.0*(q[0]*q[1] + q[2]*q[3]);

    dcm[6] = 2.0*(q[1]*q[3] + q[0]*q[2]);
    dcm[7] = 2.0*(q[2]*q[3] - q[0]*q[1]);
    dcm[8] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];

    return dcm;
}

GPisMap3::GPisMap3():t(0),
                     gpo(0),
                     obs_numdata(0)
{
    init();
}

GPisMap3::GPisMap3(GPisMap3Param par):t(0),
                                      gpo(0),
                                      obs_numdata(0),
                                      setting(par)
{
    init();
}


GPisMap3::GPisMap3(GPisMap3Param par, camParam c):t(0),
                                                  gpo(0),
                                                  obs_numdata(0),
                                                  setting(par),
                                                  cam(c)
{
    init();
}

GPisMap3::~GPisMap3()
{
    reset();
}

void GPisMap3::init(){
    pose_tr.resize(3);
    pose_R.resize(9);
}

void GPisMap3::reset(){
    if (t!=0){
        delete t;
        t = 0;
    }

    if (gpo!=0){
        delete gpo;
        gpo = 0;
    }

    obs_numdata = 0;

    activeSet.clear();

    return;
}

void GPisMap3::resetCam(camParam c){

    cam = c;
    vu_grid.clear();

    return;
}

bool GPisMap3::preprocData(FLOAT * dataz, int N, std::vector<FLOAT> & pose)
{
    if (dataz == 0 || N < 1)
        return false;

    obs_valid_xyzlocal.clear();
    obs_valid_xyzglobal.clear();
    obs_valid_u.clear();
    obs_valid_v.clear();
    obs_zinv.clear();

    range_obs_max = 0.0;

    if (pose.size() != 12)
        return false;

    std::copy(pose.begin(),pose.begin()+3,pose_tr.begin());
    std::copy(pose.begin()+3,pose.end(),pose_R.begin());

    int n = cam.width/setting.obs_skip;
    int m = cam.height/setting.obs_skip;

    // preset u- & v- grid if not done
    if (vu_grid.size() == 0)
    {
        if (cam.width*cam.height != N){
            std::cout << "Error: The dimensions do not match!" << std::endl;
            return false;
        }

        vu_grid.resize(2*n*m);

        int col = 0;
        int row = 0;
        for (int n_ = 0; n_< n; n_++){
            col = n_*setting.obs_skip;
            for (int m_ = 0; m_<m ; m_++){
                row = m_*setting.obs_skip;
                int j = 2*(m*n_+m_);
                vu_grid[j] = (FLOAT(row) - cam.cy)/cam.fy;
                vu_grid[j+1] = (FLOAT(col) - cam.cx)/cam.fx;
            }
        }

        u_obs_limit[0] = -cam.cx/cam.fx;
        u_obs_limit[1] = (FLOAT(col)- cam.cx)/cam.fx;
        v_obs_limit[0] = -cam.cy/cam.fy;
        v_obs_limit[1] = (FLOAT(row)- cam.cy)/cam.fy;
    }

    // pre-compute 3D cartesian every frame
    obs_numdata = 0;
    int temp_count = 0;
    for (int n_ = 0; n_<n ; n_++){
        int col = n_*setting.obs_skip;
        for (int m_ = 0; m_<m ; m_++){

            int row = m_*setting.obs_skip;
            int k=col*cam.height + row;

            if (  (k < N) && isRangeValid(dataz[k]) ){
                int j=2*(m*n_+m_);
                if (range_obs_max < dataz[k]) // used for range-search
                    range_obs_max = dataz[k];

                obs_zinv.push_back(1.0/dataz[k]);
                FLOAT xloc, yloc;
                FLOAT u = vu_grid[j+1];
                FLOAT v = vu_grid[j];
                obs_valid_u.push_back(u);
                obs_valid_v.push_back(v);
                xloc = u*dataz[k];
                yloc = v*dataz[k];
                obs_valid_xyzlocal.push_back(xloc);
                obs_valid_xyzlocal.push_back(yloc);
                obs_valid_xyzlocal.push_back(dataz[k]);
                obs_valid_xyzglobal.push_back(pose_R[0]*xloc + pose_R[3]*yloc + pose_R[6]*dataz[k] + pose_tr[0]);
                obs_valid_xyzglobal.push_back(pose_R[1]*xloc + pose_R[4]*yloc + pose_R[7]*dataz[k] + pose_tr[1]);
                obs_valid_xyzglobal.push_back(pose_R[2]*xloc + pose_R[5]*yloc + pose_R[8]*dataz[k] + pose_tr[2]);
                obs_numdata++;
            }
            else{
                obs_zinv.push_back(-1.0);
            }
        }
    }

    if (obs_numdata > 1)
        return true;

    return false;
}

void GPisMap3::update(FLOAT * dataz, int N, std::vector<FLOAT> & pose)
{
    if (!preprocData(dataz,N,pose))
        return;

    // Step 1
    if (regressObs()){

        // Step 2
        updateMapPoints();

        // Step 3
        addNewMeas();

        // Step 4
        updateGPs();

    }
    return;
}

bool GPisMap3::regressObs(){

    t1= std::chrono::high_resolution_clock::now();
    int dim[2];
    if (gpo == 0){
        gpo = new ObsGP2D();
    }


    if (2*obs_zinv.size() != vu_grid.size())
        return false;

    dim[0] = cam.height/setting.obs_skip;
    dim[1] = cam.width/setting.obs_skip;

    gpo->reset();
    gpo->train(vu_grid.data(), obs_zinv.data(), dim);
    t2= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_collapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1); // reset
    runtime[0] = time_collapsed.count();

    return gpo->isTrained();
}

void GPisMap3::updateMapPoints(){
    t1= std::chrono::high_resolution_clock::now();
    if (t!=0 && gpo !=0){
        AABB3 searchbb(pose_tr[0],pose_tr[1],pose_tr[2],range_obs_max);
        std::vector<OcTree*> oc;
        t->QueryNonEmptyLevelC(searchbb,oc);

        if (oc.size() > 0){

            FLOAT r2 = range_obs_max*range_obs_max;
            int k=0;
            for (auto it = oc.begin(); it != oc.end(); it++, k++) {

                Point3<FLOAT> ct = (*it)->getCenter();
                FLOAT l = (*it)->getHalfLength();
                FLOAT sqr_range = (ct.x-pose_tr[0])*(ct.x-pose_tr[0]) + (ct.y-pose_tr[1])*(ct.y-pose_tr[1])+(ct.z-pose_tr[2])*(ct.z-pose_tr[2]);

                if (sqr_range > (r2 + 2*l*l)){ // out_of_range
                    continue;
                }

                std::vector<Point3<FLOAT> > ext;
                ext.push_back((*it)->getNWF());
                ext.push_back((*it)->getNEF());
                ext.push_back((*it)->getSWF());
                ext.push_back((*it)->getSEF());
                ext.push_back((*it)->getNWB());
                ext.push_back((*it)->getNEB());
                ext.push_back((*it)->getSWB());
                ext.push_back((*it)->getSEB());

                int within_angle = 0;
                for (auto it_ = ext.begin();it_ != ext.end(); it_++){
                    FLOAT x_loc = pose_R[0]*((*it_).x-pose_tr[0])+pose_R[1]*((*it_).y-pose_tr[1])+pose_R[2]*((*it_).z-pose_tr[2]);
                    FLOAT y_loc = pose_R[3]*((*it_).x-pose_tr[0])+pose_R[4]*((*it_).y-pose_tr[1])+pose_R[5]*((*it_).z-pose_tr[2]);
                    FLOAT z_loc = pose_R[6]*((*it_).x-pose_tr[0])+pose_R[7]*((*it_).y-pose_tr[1])+pose_R[8]*((*it_).z-pose_tr[2]);
                    if (z_loc > 0){
                        FLOAT xv = x_loc/z_loc;
                        FLOAT yv = y_loc/z_loc;

                        int( (xv > u_obs_limit[0]) && (xv < u_obs_limit[1]) && ( yv > v_obs_limit[0]) && (yv < v_obs_limit[1]));
                    }

                }

                if (within_angle == 0){
                    continue;
                }

                // Get all the nodes
                std::vector<std::shared_ptr<Node3> > nodes;
                (*it)->getAllChildrenNonEmptyNodes(nodes);

                reEvalPoints(nodes);

            }
        }

    }

    t2= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_collapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1); // reset
    runtime[1] = time_collapsed.count();
    return;
}

void GPisMap3::reEvalPoints(std::vector<std::shared_ptr<Node3> >& nodes){

    // placeholders
    EMatrixX vu(2,1);
    EVectorX rinv0(1);
    EVectorX var(1);
    FLOAT ang = 0.0;
    FLOAT rinv = 0.0;

    // moving average weight
    FLOAT w = 1.0/6.0;

    // For each point
    for (auto it=nodes.begin(); it != nodes.end(); it++){

        Point3<FLOAT> pos = (*it)->getPos();

        FLOAT x_loc = pose_R[0]*(pos.x-pose_tr[0])+pose_R[1]*(pos.y-pose_tr[1])+pose_R[2]*(pos.z-pose_tr[2]);
        FLOAT y_loc = pose_R[3]*(pos.x-pose_tr[0])+pose_R[4]*(pos.y-pose_tr[1])+pose_R[5]*(pos.z-pose_tr[2]);
        FLOAT z_loc = pose_R[6]*(pos.x-pose_tr[0])+pose_R[7]*(pos.y-pose_tr[1])+pose_R[8]*(pos.z-pose_tr[2]);

        if (z_loc < 0.0)
            continue;

        vu(0) = y_loc/z_loc;
        vu(1) = x_loc/z_loc;
        rinv = 1.0/z_loc;

        gpo->test(vu,rinv0,var);

        // If unobservable, continue
        if (var(0) > setting.obs_var_thre)
            continue;


        FLOAT oc = occ_test(rinv, rinv0(0), z_loc*30.0);

        // If unobservable, continue
        if (oc < -0.1) // TO-DO : set it as a parameter
            continue;

        // gradient in the local coord.
        Point3<FLOAT> grad = (*it)->getGrad();
        FLOAT grad_loc[3];
        grad_loc[0] = pose_R[0]*grad.x + pose_R[1]*grad.y + pose_R[2]*grad.z;
        grad_loc[1] = pose_R[3]*grad.x + pose_R[4]*grad.y + pose_R[5]*grad.z;
        grad_loc[3] = pose_R[6]*grad.x + pose_R[7]*grad.y + pose_R[8]*grad.z;

        /// Compute a new position
        // Iteratively move along the normal direction.
        FLOAT abs_oc = fabs(oc);
        FLOAT dx = setting.delx;
        FLOAT x_new[3] = {x_loc, y_loc, z_loc};
        FLOAT r_new = z_loc;
        for (int i=0; i<10 && abs_oc > 0.02; i++){ // TO-DO : set it as a parameter
            // move one step
            // (the direction is determined by the occupancy sign,
            //  the step size is heuristically determined accordint to iteration.)
            if (oc < 0){
                x_new[0] += grad_loc[0]*dx;
                x_new[1] += grad_loc[1]*dx;
                x_new[2] += grad_loc[2]*dx;
            }
            else{
                x_new[0] -= grad_loc[0]*dx;
                x_new[1] -= grad_loc[1]*dx;
                x_new[2] -= grad_loc[2]*dx;
            }

            // test the new point
            vu(0) = y_loc/z_loc;
            vu(1) = x_loc/z_loc;
            r_new = z_loc;
            gpo->test(vu,rinv0,var);

            if (var(0) > setting.obs_var_thre)
                break;
            else{
                FLOAT oc_new = occ_test(1.0/(r_new), rinv0(0), r_new*30.0);
                FLOAT abs_oc_new = fabs(oc_new);

                if (abs_oc_new < 0.02 || oc < -0.1) // TO-DO : set it as a parameter
                    break;
                else if (oc*oc_new < 0.0)
                    dx = 0.5*dx; // TO-DO : set it as a parameter
                else
                    dx = 1.1*dx; // TO-DO : set it as a parameter

                abs_oc = abs_oc_new;
                oc = oc_new;
            }
        }

        // Compute its gradient and uncertainty
        FLOAT Xperturb[6] = {1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
        FLOAT Yperturb[6] = {0.0, 0.0, 1.0, -1.0, 0.0, 0.0};
        FLOAT Zperturb[6] = {0.0, 0.0, 0.0, 0.0, 1.0, -1.0};
        FLOAT occ[6] = {-1.0,-1.0,-1.0,-1.0,-1.0,-1.0};
        FLOAT occ_mean = 0.0;
        FLOAT r0_mean = 0.0;
        FLOAT r0_sqr_sum = 0.0;

        for (int i=0; i<6; i++){
            Xperturb[i] = x_new[0] + setting.delx*Xperturb[i];
            Yperturb[i] = x_new[1] + setting.delx*Yperturb[i];
            Zperturb[i] = x_new[2] + setting.delx*Zperturb[i];

            vu(0) = Yperturb[i]/Zperturb[i];
            vu(1) = Xperturb[i]/Zperturb[i];
            r_new = Zperturb[i];
            gpo->test(vu,rinv0,var);

            if (var(0) > setting.obs_var_thre)
            {
                break;
            }
            occ[i] = occ_test(1.0/r_new, rinv0(0), r_new*30.0);
            occ_mean += w*occ[i];
            FLOAT r0 = 1.0/rinv0(0);
            r0_sqr_sum += r0*r0;
            r0_mean += w*r0;
        }


        if (var(0) > setting.obs_var_thre)// invalid
            continue;

        Point3<FLOAT> grad_new_loc,grad_new;

        grad_new_loc.x = (occ[0] -occ[1])/setting.delx;
        grad_new_loc.y = (occ[2] -occ[3])/setting.delx;
        grad_new_loc.z = (occ[4] -occ[5])/setting.delx;
        FLOAT norm_grad_new = std::sqrt(grad_new_loc.x*grad_new_loc.x + grad_new_loc.y*grad_new_loc.y + grad_new_loc.z*grad_new_loc.z);

        if (norm_grad_new <1e-3){ // uncertainty increased
            (*it)->updateNoise(2.0*(*it)->getPosNoise(),2.0*(*it)->getGradNoise());
            continue;
        }

        FLOAT r_var = r0_sqr_sum/5.0 - r0_mean*r0_mean*6.0/5.0;
        r_var /= setting.delx;
        FLOAT noise = 100.0;
        FLOAT grad_noise = 1.0;
        if (norm_grad_new > 1e-6){
            grad_new_loc.x = grad_new_loc.x/norm_grad_new;
            grad_new_loc.y = grad_new_loc.y/norm_grad_new;
            grad_new_loc.z = grad_new_loc.z/norm_grad_new;
            noise = setting.min_position_noise*saturate(r_new*r_new, 1.0, noise);
            grad_noise = saturate(std::fabs(occ_mean)+r_var,setting.min_grad_noise,grad_noise);
        }
        else{
            noise = setting.min_position_noise*noise;
        }


        FLOAT dist = std::sqrt(x_new[0]*x_new[0]+x_new[1]*x_new[1]+x_new[2]*x_new[2]);
        FLOAT view_ang = std::max(-(x_new[0]*grad_new_loc.x+x_new[1]*grad_new_loc.y+x_new[2]*grad_new_loc.z)/dist, (FLOAT)1e-1);
        FLOAT view_ang2 = view_ang*view_ang;
        FLOAT view_noise = setting.min_position_noise*((1.0-view_ang2)/view_ang2);

        FLOAT temp = noise;
        noise += view_noise + abs_oc;
        grad_noise = grad_noise + 0.1*view_noise;


        // local to global coord.
        Point3<FLOAT> pos_new;
        pos_new.x = pose_R[0]*x_new[0] + pose_R[3]*x_new[1] + pose_R[6]*x_new[2] + pose_tr[0];
        pos_new.y = pose_R[1]*x_new[0] + pose_R[4]*x_new[1] + pose_R[7]*x_new[2] + pose_tr[1];
        pos_new.z = pose_R[2]*x_new[0] + pose_R[5]*x_new[1] + pose_R[8]*x_new[2] + pose_tr[2];
        grad_new.x = pose_R[0]*grad_new_loc.x + pose_R[3]*grad_new_loc.y + pose_R[6]*grad_new_loc.z;
        grad_new.y = pose_R[1]*grad_new_loc.x + pose_R[4]*grad_new_loc.y + pose_R[7]*grad_new_loc.z;
        grad_new.z = pose_R[2]*grad_new_loc.x + pose_R[5]*grad_new_loc.y + pose_R[8]*grad_new_loc.z;

        FLOAT noise_old = (*it)->getPosNoise();
        FLOAT grad_noise_old = (*it)->getGradNoise();

        FLOAT pos_noise_sum = (noise_old + noise);
        FLOAT grad_noise_sum = (grad_noise_old + grad_noise);


         // Now, update
        if (grad_noise_old > 0.5 || grad_noise_old > 0.6){
           ;
        }
        else{
            // Position update
            pos_new.x = (noise*pos.x + noise_old*pos_new.x)/pos_noise_sum;
            pos_new.y = (noise*pos.y + noise_old*pos_new.y)/pos_noise_sum;
            pos_new.z = (noise*pos.z + noise_old*pos_new.z)/pos_noise_sum;
            FLOAT dist = 0.5*std::sqrt((pos.x-pos_new.x)*(pos.x-pos_new.x) + (pos.y-pos_new.y)*(pos.y-pos_new.y) + (pos.z-pos_new.z)*(pos.z-pos_new.z));

            // Normal update
            Point3<FLOAT> axis; // cross product
            axis.x =  grad_new.y*grad.z - grad_new.z*grad.y;
            axis.y = -grad_new.x*grad.z + grad_new.z*grad.x;
            axis.z =  grad_new.x*grad.y - grad_new.y*grad.x;
            FLOAT ang = acos(grad_new.x*grad.x+grad_new.y*grad.y+grad_new.z*grad.z);
            ang = ang*noise/pos_noise_sum;
            // rotvect
            FLOAT q[4] = {1.0, 0.0, 0.0, 0.0};
            if (ang > 1-6){
                q[0] = cos(ang/2.0);
                FLOAT sina = sin(ang/2.0);
                q[1] = axis.x*sina;
                q[2] = axis.y*sina;
                q[3] = axis.z*sina;
            }
            // quat 2 dcm
            std::array<FLOAT,9> Rot = quat2dcm(q);

            grad_new.x = Rot[0]*grad.x + Rot[1]*grad.y + Rot[2]*grad.z;
            grad_new.y = Rot[3]*grad.x + Rot[4]*grad.y + Rot[5]*grad.z;
            grad_new.z = Rot[6]*grad.x + Rot[7]*grad.y + Rot[8]*grad.z;

            // Noise update
            grad_noise = std::min((FLOAT)1.0, std::max(grad_noise*grad_noise_old/grad_noise_sum + dist, setting.map_noise_param));
            noise = std::max((noise*noise_old/pos_noise_sum + dist), setting.map_noise_param);
        }
        // remove
        t->Remove(*it,activeSet);

        if (noise > 1.0 && grad_noise > 0.61){
            continue;
        }
        else{
            // try inserting
            std::shared_ptr<Node3> p(new Node3(pos_new));
            std::unordered_set<OcTree*> vecInserted;

            bool succeeded = false;
            if (!t->IsNotNew(p)){
                succeeded = t->Insert(p,vecInserted);
                if (succeeded){
                    if (t->IsRoot() == false){
                        t = t->getRoot();
                    }
                }
            }

            if ((succeeded == 0) || !(vecInserted.size() > 0)) // if failed, then continue to test the next point
                continue;

            // update the point
            p->updateData(-setting.fbias, noise, grad_new, grad_noise, NODE_TYPE::HIT);

            for (auto itv = vecInserted.begin(); itv != vecInserted.end(); itv++)
                activeSet.insert(*itv);
        }

    }

    return;
}

void GPisMap3::addNewMeas(){
    // create if not initialized
    if (t == 0){
        t = new OcTree(Point3<FLOAT>(0.0,0.0,0.0));
    }
    evalPoints();
    return;
}

void GPisMap3::evalPoints(){

    t1= std::chrono::high_resolution_clock::now();
     if (t == 0 || obs_numdata < 1)
         return;

    FLOAT w = 1.0/6.0;
    int k=0;
    // For each point
    for (; k<obs_numdata; k++){
        int k2 = 2*k;
        int k3 = 3*k;

        // placeholder;
        EVectorX rinv0(1);
        EVectorX var(1);
        EMatrixX vu(2,1);

        vu(0,0) = obs_valid_v[k];
        vu(1,0) = obs_valid_u[k];
        gpo->test(vu,rinv0,var);

        if (var(0) > setting.obs_var_thre)
        {
            continue;
        }

        /////////////////////////////////////////////////////////////////
        // Try inserting
        Point3<FLOAT> pt(obs_valid_xyzglobal[k3],obs_valid_xyzglobal[k3+1],obs_valid_xyzglobal[k3+2]);
        std::shared_ptr<Node3> p(new Node3(pt));
        std::unordered_set<OcTree*> vecInserted;

        bool succeeded = false;
        if (!t->IsNotNew(p)){
            succeeded = t->Insert(p,vecInserted);
            if (succeeded){
                if (t->IsRoot() == false){
                    t = t->getRoot();
                }
            }
        }


        if ((succeeded == 0) || !(vecInserted.size() > 0)) // if failed, then continue to test the next point
            continue;

        /////////////////////////////////////////////////////////////////
        // if succeeded, then compute surface normal and uncertainty
        FLOAT Xperturb[6] = {1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
        FLOAT Yperturb[6] = {0.0, 0.0, 1.0, -1.0, 0.0, 0.0};
        FLOAT Zperturb[6] = {0.0, 0.0, 0.0, 0.0, 1.0, -1.0};
        FLOAT occ[6] = {-1.0,-1.0,-1.0,-1.0,-1.0,-1.0};
        FLOAT occ_mean = 0.0;
        int i=0;
        for (; i<6; i++){
            Xperturb[i] = obs_valid_xyzlocal[k3] + setting.delx*Xperturb[i];
            Yperturb[i] = obs_valid_xyzlocal[k3+1] + setting.delx*Yperturb[i];
            Zperturb[i] = obs_valid_xyzlocal[k3+2] + setting.delx*Zperturb[i];


            vu(0,0) = Yperturb[i]/Zperturb[i];
            vu(1,0) = Xperturb[i]/Zperturb[i];
           // std::cout << vu << std::endl;
            gpo->test(vu,rinv0,var);

            if (var(0) > setting.obs_var_thre)
            {
                break;
            }
            occ[i] = occ_test(1.0/Zperturb[i], rinv0(0), Zperturb[i]*30.0);
            occ_mean += w*occ[i];
        }

        if (var(0) > setting.obs_var_thre){
            //std::cout << "Removing...." << var(0) << std::endl;
            t->Remove(p);
            continue;
        }


        FLOAT noise = 100.0;
        FLOAT grad_noise = 1.00;
        Point3<FLOAT> grad;

        grad.x = (occ[0] -occ[1])/setting.delx;
        grad.y = (occ[2] -occ[3])/setting.delx;
        grad.z = (occ[4] -occ[5])/setting.delx;
        FLOAT norm_grad = grad.x*grad.x + grad.y*grad.y + grad.z*grad.z;

        if (norm_grad > 1e-6){
            norm_grad = std::sqrt(norm_grad);
            FLOAT grad_loc_x = grad.x/norm_grad;
            FLOAT grad_loc_y = grad.y/norm_grad;
            FLOAT grad_loc_z = grad.z/norm_grad;

            grad.x = pose_R[0]*grad_loc_x + pose_R[3]*grad_loc_y + pose_R[6]*grad_loc_z;
            grad.y = pose_R[1]*grad_loc_x + pose_R[4]*grad_loc_y + pose_R[7]*grad_loc_z;
            grad.z = pose_R[2]*grad_loc_x + pose_R[5]*grad_loc_y + pose_R[8]*grad_loc_z;

            FLOAT dist = std::sqrt(obs_valid_xyzlocal[k3]*obs_valid_xyzlocal[k3]+obs_valid_xyzlocal[k3+1]*obs_valid_xyzlocal[k3+1]+obs_valid_xyzlocal[k3+2]*obs_valid_xyzlocal[k3+2]);
            noise = setting.min_position_noise*(saturate(dist, 1.0, noise));
            grad_noise = saturate(std::fabs(occ_mean),setting.min_grad_noise,grad_noise);

            FLOAT view_ang = std::max(-(obs_valid_xyzlocal[k3]*grad_loc_x+obs_valid_xyzlocal[k3+1]*grad_loc_y +obs_valid_xyzlocal[k3+2]*grad_loc_z)/dist, (FLOAT)1e-1);
            FLOAT view_ang2 = view_ang*view_ang;
            FLOAT view_noise = setting.min_position_noise*((1.0-view_ang2)/view_ang2);
            noise += view_noise;
        }

        /////////////////////////////////////////////////////////////////
        // update the point
        p->updateData(-setting.fbias, noise, grad,grad_noise, NODE_TYPE::HIT);

        for (auto it = vecInserted.begin(); it != vecInserted.end(); it++){
            activeSet.insert(*it);
        }

    }

    t2= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_collapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1); // reset
    runtime[2] = time_collapsed.count();
    return;
}

void GPisMap3::updateGPs_kernel(int thread_idx,
                                int start_idx,
                                int end_idx,
                                OcTree **nodes_to_update){
    std::vector<std::shared_ptr<Node3> > res;
    for (int i = start_idx; i < end_idx; ++i){
        if (nodes_to_update[i] != 0){
            Point3<FLOAT> ct = (nodes_to_update[i])->getCenter();
            FLOAT l = (nodes_to_update[i])->getHalfLength();
            AABB3 searchbb(ct.x,ct.y,ct.z, l*Rtimes);
            res.clear();
            t->QueryRange(searchbb,res);
            if (res.size()>0){
                std::shared_ptr<OnGPIS> gp(new OnGPIS(setting.map_scale_param, setting.map_noise_param));
                gp->train(res);
                (nodes_to_update[i])->Update(gp);
            }
        }
    }

}

void GPisMap3::updateGPs(){
    t1= std::chrono::high_resolution_clock::now();

    std::unordered_set<OcTree*> updateSet(activeSet);

    int num_threads = std::thread::hardware_concurrency();
    std::thread *threads = new std::thread[num_threads];

    int num_threads_to_use = num_threads;

    for (auto it = activeSet.begin(); it!= activeSet.end(); it++){

        Point3<FLOAT> ct = (*it)->getCenter();
        FLOAT l = (*it)->getHalfLength();
        AABB3 searchbb(ct.x,ct.y,ct.z, Rtimes*l);
        std::vector<OcTree*> qs;
        t->QueryNonEmptyLevelC(searchbb,qs);
        if (qs.size()>0){
            for (auto itq = qs.begin(); itq!=qs.end(); itq++){
                updateSet.insert(*itq);
            }
        }
    }

    int num_elements = updateSet.size();
    OcTree **nodes_to_update = new OcTree*[num_elements];
    int it_counter = 0;
    for (auto it = updateSet.begin(); it != updateSet.end(); ++it, ++it_counter){
        nodes_to_update[it_counter] = *it;
    }

    if (num_elements < num_threads){
        num_threads_to_use = num_elements;
    }
    else{
        num_threads_to_use = num_threads;
    }
    int num_leftovers = num_elements % num_threads_to_use;
    int batch_size = num_elements / num_threads_to_use;
    int element_cursor = 0;
    for(int i = 0; i < num_leftovers; ++i){
        threads[i] = std::thread(&GPisMap3::updateGPs_kernel,
                                 this,
                                 i,
                                 element_cursor,
                                 element_cursor + batch_size + 1,
                                 nodes_to_update);
        element_cursor += batch_size + 1;

    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i){
        threads[i] = std::thread(&GPisMap3::updateGPs_kernel,
                                 this,
                                 i,
                                 element_cursor,
                                 element_cursor + batch_size,
                                 nodes_to_update);
        element_cursor += batch_size;
    }

    for (int i = 0; i < num_threads_to_use; ++i){
        threads[i].join();
    }

    delete [] nodes_to_update;
    delete [] threads;
    // clear active set once all the jobs for update are done.
    activeSet.clear();
    t2= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_collapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1); // reset
    runtime[3] = time_collapsed.count();
    return;
}

void GPisMap3::test_kernel(int thread_idx,
                           int start_idx,
                           int end_idx,
                           FLOAT *x,
                           FLOAT *res){

    FLOAT var_thre = 0.5; // TO-DO

    for(int i = start_idx; i < end_idx; ++i){

        int k3 = 3*i;
        EVectorX xt(3);
        xt << x[k3], x[k3+1], x[k3+2];

        int k8 = 8*i;

        // query Cs
        AABB3 searchbb(xt(0),xt(1),xt(2),C_leng*3.0);
        std::vector<OcTree*> quads;
        std::vector<FLOAT> sqdst;
        t->QueryNonEmptyLevelC(searchbb,quads,sqdst);

        res[k8+4] = 1.0 + setting.map_noise_param ; // variance of sdf value

        if (quads.size() == 1){
            std::shared_ptr<OnGPIS> gp = quads[0]->getGP();
            if (gp != nullptr){
                gp->testSinglePoint(xt,res[k8],&res[k8+1],&res[k8+4]);
            }
        }
        else if (sqdst.size() > 1){
            // sort by distance
            std::vector<int> idx(sqdst.size());
            std::size_t n(0);
            std::generate(std::begin(idx), std::end(idx), [&]{ return n++; });
            std::sort(  std::begin(idx), std::end(idx),[&](int i1, int i2) { return sqdst[i1] < sqdst[i2]; } );

            // get THE FIRST gp pointer
            std::shared_ptr<OnGPIS> gp = quads[idx[0]]->getGP();
            if (gp != nullptr){
                gp->testSinglePoint(xt,res[k8],&res[k8+1],&res[k8+4]);
            }

            if (res[k8+4] > var_thre){

                FLOAT f2[8];
                FLOAT grad2[8*3];
                FLOAT var2[8*4];

                var2[0] = res[k8+4];
                int numc = sqdst.size();
                if (numc > 3) numc = 3;
                bool need_wsum = true;

                for (int m=0; m<(numc-1);m++){
                    int m_1 = m+1;
                    int m3 = m_1*3;
                    int m4 = m_1*4;
                    gp = quads[idx[m_1]]->getGP();
                    gp->testSinglePoint(xt,f2[m_1],&grad2[m3],&var2[m4]);
                }

                if (need_wsum){
                    f2[0] = res[k8];
                    grad2[0] = res[k8+1];
                    grad2[1] = res[k8+2];
                    grad2[2] = res[k8+3];
                    var2[1] = res[k8+5];
                    var2[2] = res[k8+6];
                    var2[3] = res[k8+7];
                    std::vector<int> idx(numc);
                    std::size_t n(0);
                    std::generate(std::begin(idx), std::end(idx), [&]{ return n++; });
                    std::sort(  std::begin(idx), std::end(idx),[&](int i1, int i2) { return var2[i1*4] < var2[i2*4]; } );

                    if (var2[idx[0]*4] < var_thre)
                    {
                        res[k8] = f2[idx[0]];
                        res[k8+1] = grad2[idx[0]*3];
                        res[k8+2] = grad2[idx[0]*3+1];
                        res[k8+3] = grad2[idx[0]*3+2];

                        res[k8+4] = var2[idx[0]*4];
                        res[k8+5] = var2[idx[0]*4+1];
                        res[k8+6] = var2[idx[0]*4+2];
                        res[k8+7] = var2[idx[0]*4+3];
                    }
                    else{
                        FLOAT w1 = (var2[idx[0]*4] - var_thre);
                        FLOAT w2 = (var2[idx[1]*4]- var_thre);
                        FLOAT w12 = w1+w2;

                        res[k8] = (w2*f2[idx[0]]+w1*f2[idx[1]])/w12;
                        res[k8+1] = (w2*grad2[idx[0]*3]+w1*grad2[idx[1]*3])/w12;
                        res[k8+2] = (w2*grad2[idx[0]*3+1]+w1*grad2[idx[1]*3+1])/w12;
                        res[k8+3] = (w2*grad2[idx[0]*3+2]+w1*grad2[idx[1]*3+2])/w12;

                        res[k8+4] = (w2*var2[idx[0]*4]+w1*var2[idx[1]*4])/w12;
                        res[k8+5] = (w2*var2[idx[0]*4+1]+w1*var2[idx[1]*4+1])/w12;
                        res[k8+6] = (w2*var2[idx[0]*4+2]+w1*var2[idx[1]*4+2])/w12;
                        res[k8+7] = (w2*var2[idx[0]*4+3]+w1*var2[idx[1]*4+3])/w12;
                    }
                }
            }
        }

    }

}

bool GPisMap3::test(FLOAT * x, int dim, int leng, FLOAT * res){
    if (x == 0 || dim != mapDimension || leng < 1)
        return false;

    int num_threads = std::thread::hardware_concurrency();
    int num_threads_to_use = num_threads;
    if (leng < num_threads){
        num_threads_to_use = leng;
    }
    else{
        num_threads_to_use = num_threads;
    }
    std::thread *threads = new std::thread[num_threads_to_use];

    int num_leftovers = leng % num_threads_to_use;
    int batch_size = leng / num_threads_to_use;
    int element_cursor = 0;

    for(int i = 0; i < num_leftovers; ++i){
        threads[i] = std::thread(&GPisMap3::test_kernel,
                                 this,
                                 i,
                                 element_cursor,
                                 element_cursor + batch_size + 1,
                                 x, res);
        element_cursor += batch_size + 1;

    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i){
        threads[i] = std::thread(&GPisMap3::test_kernel,
                                 this,
                                 i,
                                 element_cursor,
                                 element_cursor + batch_size,
                                 x, res);
        element_cursor += batch_size;
    }

    for (int i = 0; i < num_threads_to_use; ++i){
        threads[i].join();
    }

    delete [] threads;

    return true;
}

void GPisMap3::getAllPoints(std::vector<FLOAT> & pos)
{
    pos.clear();

    if (t==0)
        return;

    std::vector<std::shared_ptr<Node3> > nodes;
    t->getAllChildrenNonEmptyNodes(nodes);

    int N = nodes.size();
    if (N> 0){
        pos.resize(3*N);
        for (int j=0; j<N; j++) {
            int j3 = 3*j;
            pos[j3] = nodes[j]->getPosX();
            pos[j3+1] = nodes[j]->getPosY();
            pos[j3+2] = nodes[j]->getPosZ();
        }
    }
    return;
}

