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

#ifndef __OCTREE_H_
#define __OCTREE_H_

#include <vector>
#include <map>
#include <unordered_set>
#include <memory>
#include <iostream>
#include <cstdint>
#include "strct.h"
#include "OnGPIS.h"

class AABB3
{
    Point3<float> center;
    float halfLength;
    float halfLengthSq;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;

    Point3<float> ptNWF;
    Point3<float> ptNEF;
    Point3<float> ptSWF;
    Point3<float> ptSEF;
    Point3<float> ptNWB;
    Point3<float> ptNEB;
    Point3<float> ptSWB;
    Point3<float> ptSEB;
public:
    AABB3(){
        halfLength = 0.0;
        halfLengthSq = 0.0;
        xmin = 0.0;
        xmax = 0.0;
        ymin = 0.0;
        ymax = 0.0;
        zmin = 0.0;
        zmax = 0.0;
    }
    AABB3(Point3<float> _center, float _halfLength) {
        center = _center;
        halfLength = _halfLength;
        halfLengthSq = halfLength*halfLength;
        xmin = center.x - halfLength;
        xmax = center.x + halfLength;
        ymin = center.y - halfLength;
        ymax = center.y + halfLength;
        zmin = center.z - halfLength;
        zmax = center.z + halfLength;
        ptNWF = Point3<float>(xmin,ymax,zmax);
        ptNEF = Point3<float>(xmax,ymax,zmax);
        ptSWF = Point3<float>(xmin,ymin,zmax);
        ptSEF = Point3<float>(xmax,ymin,zmax);
        ptNWB = Point3<float>(xmin,ymax,zmin);
        ptNEB = Point3<float>(xmax,ymax,zmin);
        ptSWB = Point3<float>(xmin,ymin,zmin);
        ptSEB = Point3<float>(xmax,ymin,zmin);
    }
    AABB3(float x, float y, float z, float _halfLength) {
        center = Point3<float>(x,y,z);
        halfLength = _halfLength;
        halfLengthSq = halfLength*halfLength;
        xmin = center.x - halfLength;
        xmax = center.x + halfLength;
        ymin = center.y - halfLength;
        ymax = center.y + halfLength;
        zmin = center.z - halfLength;
        zmax = center.z + halfLength;
        ptNWF = Point3<float>(xmin,ymax,zmax);
        ptNEF = Point3<float>(xmax,ymax,zmax);
        ptSWF = Point3<float>(xmin,ymin,zmax);
        ptSEF = Point3<float>(xmax,ymin,zmax);
        ptNWB = Point3<float>(xmin,ymax,zmin);
        ptNEB = Point3<float>(xmax,ymax,zmin);
        ptSWB = Point3<float>(xmin,ymin,zmin);
        ptSEB = Point3<float>(xmax,ymin,zmin);
    }

    const Point3<float> getCenter(){return center;}
    float getHalfLength(){return halfLength;}
    float getHalfLengthSq(){return halfLengthSq;}
    float getXMinbound(){return xmin;}
    float getXMaxbound(){return xmax;}
    float getYMinbound(){return ymin;}
    float getYMaxbound(){return ymax;}
    float getZMinbound(){return zmin;}
    float getZMaxbound(){return zmax;}
    const Point3<float>& getNWF(){return ptNWF;}
    const Point3<float>& getNEF(){return ptNEF;}
    const Point3<float>& getSWF(){return ptSWF;}
    const Point3<float>& getSEF(){return ptSEF;}
    const Point3<float>& getNWB(){return ptNWB;}
    const Point3<float>& getNEB(){return ptNEB;}
    const Point3<float>& getSWB(){return ptSWB;}
    const Point3<float>& getSEB(){return ptSEB;}

    bool containsPoint(Point3<float> pt) {
        return ((pt.x > xmin) &&
            (pt.x < xmax) &&
            (pt.y > ymin) &&
            (pt.y < ymax) &&
            (pt.z > zmin) &&
            (pt.z < zmax)) ;
    }

    bool intersectsAABB(AABB3 aabb) {
        return !((aabb.getXMaxbound() < xmin) ||
                (aabb.getXMinbound() > xmax) ||
                (aabb.getYMaxbound() < ymin) ||
                (aabb.getYMinbound() > ymax) ||
                (aabb.getZMaxbound() < zmin) ||
                (aabb.getZMinbound() > zmax) );
    }
};

enum class OctChildType {undefined, NWF, NEF, SWF, SEF, NWB, NEB, SWB, SEB};

class OcTree
{
    // Axis-aligned bounding box stored as a center with half-dimensionss
    // to represent the boundaries of this oct tree
    AABB3 boundary;

    static tree_param param;    // see strct.h for definition

    // Points in this oct tree node
    std::shared_ptr<Node3> node;
    std::shared_ptr<OnGPIS> gp;

    bool leaf;
    bool maxDepthReached;
    bool rootLimitReached;
    bool clevel;

    // Children
    OcTree* northWestFront;
    OcTree* northEastFront;
    OcTree* southWestFront;
    OcTree* southEastFront;
    OcTree* northWestBack;
    OcTree* northEastBack;
    OcTree* southWestBack;
    OcTree* southEastBack;

    OcTree* par;

    std::map<OctChildType, OcTree*> children_map;
    std::map<OctChildType, Point3<float>> children_center;

    void Subdivide(); // create four children that fully divide this oct into four octs of equal area
    void SubdivideExcept(OctChildType childType);
    void deleteChildren();
    void deleteNode();
    void deleteGP();
    void resetChildrenMap();
    void updateChildrenCenter();
    bool InsertToParent(std::shared_ptr<Node3> n);  
    void setParent(OcTree* const p){par = p;}

    OcTree(AABB3 _boundary, OcTree* const p = nullptr );
    OcTree(AABB3 _boundary, OcTree* const ch, OctChildType child_type);

protected:
    OcTree* const getParent(){return par;}
    bool IsLeaf(){ return leaf;} // leaf is true if chlidren are initialized
    bool IsEmpty(){return (node == nullptr);} // empty if the data node is null
    bool IsEmptyLeaf(){
        return ( leaf & (node == nullptr)); // true if no data node no child
    }
public:
    // Methods
    OcTree():northWestFront(nullptr),
            northEastFront(nullptr),
            southWestFront(nullptr),
            southEastFront(nullptr),
            northWestBack(nullptr),
            northEastBack(nullptr),
            southWestBack(nullptr),
            southEastBack(nullptr),
            par(nullptr),
            maxDepthReached(false),
            rootLimitReached(false),
            leaf(true),
            node(nullptr),
            gp(nullptr){}

    OcTree(Point3<float> center);
    ~OcTree();

    bool IsRoot(){
        if (par)
            return false;
        else
            return true;
    }

    OcTree* const getRoot();
    bool IsCluster(){return clevel;}

    bool Insert(std::shared_ptr<Node3> n);
    bool Insert(std::shared_ptr<Node3> n, std::unordered_set<OcTree*>& octs);
    bool IsNotNew(std::shared_ptr<Node3> n);
    bool Update(std::shared_ptr<Node3> n);
    bool Update(std::shared_ptr<Node3> n, std::unordered_set<OcTree*>& octs);
    bool Remove(std::shared_ptr<Node3> n, std::unordered_set<OcTree*>& octs);

    void InitGP(float scale_param, float noise_param);
    void UpdateGP(const vecNode3& samples);
    std::shared_ptr<OnGPIS> const getGP(){return gp;}
    
    bool Remove(std::shared_ptr<Node3> n);
    void QueryRange(AABB3 range, std::vector<std::shared_ptr<Node3> >& nodes);
    void QueryNonEmptyLevelC(AABB3 range, std::vector<OcTree*>& octs);
    void QueryNonEmptyLevelC(AABB3 range, std::vector<OcTree*>& octs, std::vector<float>& sqdst);
    void QueryNonEmptyLevelC(AABB3 range, std::vector<OcTree*>& octs, std::vector<std::vector<std::shared_ptr<Node3> > >& nodes);

    int32_t getNodeCount();
    Point3<float> getCenter(){return boundary.getCenter();}
    float getHalfLength(){return boundary.getHalfLength();}
    float getXMaxbound(){return boundary.getXMaxbound();}
    float getXMinbound(){return boundary.getXMinbound();}
    float getYMaxbound(){return boundary.getYMaxbound();}
    float getYMinbound(){return boundary.getYMinbound();}
    float getZMinbound(){return boundary.getZMinbound();}
    float getZMaxbound(){return boundary.getZMaxbound();}
    Point3<float> getChildCenter(OctChildType c);
    std::map<OctChildType, Point3<float>> const & getAllChildrenCenter();
    std::map<OctChildType, OcTree*> const & getAllChildren();

    void getAllChildrenNonEmptyNodes(std::vector<std::shared_ptr<Node3> >& nodes);
    void getChildNonEmptyNodes(OctChildType c, std::vector<std::shared_ptr<Node3> >& nodes);

};

#endif
