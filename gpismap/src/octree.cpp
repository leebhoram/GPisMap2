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

#include "octree.h"

#define EPS 1e-12

static float sqdist(const Point3<float>& pt1, const Point3<float>& pt2)
{
    float dx = (pt1.x - pt2.x);
    float dy = (pt1.y - pt2.y);
    float dz = (pt1.z - pt2.z);

    return dx*dx + dy*dy + dz*dz;
}

OcTree::OcTree(Point3<float> c)
        :northWestFront(nullptr),
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
        clevel(false),
        node(nullptr),
        gp(nullptr){
    boundary = AABB3(c,OcTree::param.initroot_halfleng);
    resetChildrenMap();
}

OcTree::OcTree(AABB3 _boundary, OcTree* const p )
        :northWestFront(nullptr),
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
        clevel(false),
        node(nullptr),
        gp(nullptr){
    boundary = _boundary;
    if (boundary.getHalfLength() < OcTree::param.min_halfleng)
        maxDepthReached = true;
    if (boundary.getHalfLength() > OcTree::param.max_halfleng)
        rootLimitReached = true;
    if (fabs(getHalfLength()-OcTree::param.cluster_halfleng) < EPS)
        clevel = true;
    if (p!=nullptr)
        par = p;
    resetChildrenMap();
}

OcTree::OcTree(AABB3 _boundary,  OcTree* const ch,  OctChildType child_type)
        :northWestFront(nullptr),
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
        clevel(false),
        node(nullptr),
        gp(nullptr){
    boundary = _boundary;
    if (boundary.getHalfLength() < OcTree::param.min_halfleng)
        maxDepthReached = true;
    if (boundary.getHalfLength() > OcTree::param.max_halfleng)
        rootLimitReached = true;
    if (fabs(getHalfLength()-OcTree::param.cluster_halfleng) < EPS)
        clevel = true;
    if (child_type == OctChildType::undefined)
    {
        leaf = true;
    }
    else
    {
        leaf = false;
        SubdivideExcept(child_type);
        if (child_type == OctChildType::NWF)
            northWestFront = ch;
        if (child_type == OctChildType::NEF)
            northEastFront = ch;
        if (child_type == OctChildType::SWF)
            southWestFront = ch;
        if (child_type == OctChildType::SEF)
            southEastFront = ch;
        if (child_type == OctChildType::NWB)
            northWestBack = ch;
        if (child_type == OctChildType::NEB)
            northEastBack = ch;
        if (child_type == OctChildType::SWB)
            southWestBack = ch;
        if (child_type == OctChildType::SEB)
            southEastBack = ch;
    }
    resetChildrenMap();
}

OcTree::~OcTree(){
    deleteChildren();
    deleteGP();
    deleteNode();
}

void OcTree::deleteNode(){
    if (node != nullptr){
        node.reset();
        node = nullptr;
    }
}

void OcTree::deleteGP(){
    if (gp != nullptr){
        gp.reset();
        gp = nullptr;
    }
}

void OcTree::deleteChildren()
{
    if (northWestFront) {delete northWestFront; northWestFront = nullptr;}
    if (northEastFront) {delete northEastFront; northEastFront = nullptr;}
    if (southWestFront) {delete southWestFront; southWestFront = nullptr;}
    if (southEastFront) {delete southEastFront; southEastFront = nullptr;}
    if (northWestBack) {delete northWestBack; northWestBack = nullptr;}
    if (northEastBack) {delete northEastBack; northEastBack = nullptr;}
    if (southWestBack) {delete southWestBack; southWestBack = nullptr;}
    if (southEastBack) {delete southEastBack; southEastBack = nullptr;}
    leaf = true;
    resetChildrenMap();
}

void OcTree::resetChildrenMap(){
    children_map.clear();
    children_map.insert({OctChildType::NEB, northEastBack});
    children_map.insert({OctChildType::NWB, northWestBack});
    children_map.insert({OctChildType::SEB, southEastBack});
    children_map.insert({OctChildType::SWB, southWestBack});
    children_map.insert({OctChildType::NEF, northEastFront});
    children_map.insert({OctChildType::NWF, northWestFront});
    children_map.insert({OctChildType::SEF, southEastFront});
    children_map.insert({OctChildType::SWF, southWestFront});
    updateChildrenCenter();
}

void OcTree::updateChildrenCenter(){
    children_center.clear();
    for (auto &child: children_map)
        if (child.second!=nullptr){
            children_center.insert({child.first, child.second->getCenter()});
        }
}

std::map<OctChildType, Point3<float>> const & OcTree::getAllChildrenCenter()
{
    return children_center;
}

std::map<OctChildType, OcTree*> const & OcTree::getAllChildren()
{   
    return children_map;
}

OcTree* const OcTree::getRoot(){
    OcTree* p = this;
    OcTree* p1 = p->getParent();
    while (p1!=nullptr){
        p = p1;
        p1 = p->getParent();
    }
    return p;
}

void OcTree::InitGP(float scale_param, float noise_param)
{
    deleteGP();
    gp = std::make_shared<OnGPIS>(scale_param, noise_param);
}

void OcTree::UpdateGP(const vecNode3& samples)
{
    if (gp != nullptr)
        gp->train(samples);
}

bool OcTree::InsertToParent(std::shared_ptr<Node3> n){
    float l = getHalfLength();
    Point3<float> c = getCenter();

    // Find out what type the current node is
    const Point3<float> np = n->getPos();

    Point3<float> par_c;
    OctChildType childType = OctChildType::undefined;
    if (np.x < c.x && np.y > c.y && np.z > c.z){
        childType = OctChildType::SEB;
        par_c.x = c.x - l;
        par_c.y = c.y + l;
        par_c.z = c.z + l;
    }
    if (np.x > c.x && np.y > c.y && np.z > c.z){
        childType = OctChildType::SWB;
        par_c.x = c.x + l;
        par_c.y = c.y + l;
        par_c.z = c.z + l;
    }
    if (np.x < c.x && np.y < c.y && np.z > c.z){
        childType = OctChildType::NEB;
        par_c.x = c.x - l;
        par_c.y = c.y - l;
        par_c.z = c.z + l;
    }
    if (np.x > c.x && np.y < c.y && np.z > c.z){
        childType = OctChildType::NWB;
        par_c.x = c.x + l;
        par_c.y = c.y - l;
        par_c.z = c.z + l;
    }
    if (np.x < c.x && np.y > c.y && np.z < c.z){
        childType = OctChildType::SEF;
        par_c.x = c.x - l;
        par_c.y = c.y + l;
        par_c.z = c.z - l;
    }
    if (np.x > c.x && np.y > c.y && np.z < c.z){
        childType = OctChildType::SWF;
        par_c.x = c.x + l;
        par_c.y = c.y + l;
        par_c.z = c.z - l;
    }
    if (np.x < c.x && np.y < c.y && np.z < c.z){
        childType = OctChildType::NEF;
        par_c.x = c.x - l;
        par_c.y = c.y - l;
        par_c.z = c.z - l;
    }
    if (np.x > c.x && np.y < c.y && np.z < c.z){
        childType = OctChildType::NWF;
        par_c.x = c.x + l;
        par_c.y = c.y - l;
        par_c.z = c.z - l;
    }

    AABB3 parbb(par_c,2.0*l);
    par = new OcTree(parbb,this,childType);
    return par->Insert(n);
}

bool OcTree::Insert(std::shared_ptr<Node3> n){

    // Ignore objects that do not belong in this oct tree
    if (!boundary.containsPoint(n->getPos())){
        if (getParent() == nullptr) {
            if (rootLimitReached)  return false;
            else                   return InsertToParent(n);
        }
        return false; // object cannot be added
    }

    if (maxDepthReached){
        if (IsEmpty()) {// If this is the first point in this oct tree, add the object here
            node = n;
            return true;
        }
        else // no more points accepted at this resolution
            return false;
    }

    if (leaf){

        if (boundary.getHalfLength() > OcTree::param.cluster_halfleng){
            Subdivide();
        }
        else{
            if (IsEmpty())
            // If this is the first point in this oct tree, add the object here
            {
                node = n;
                return true;
            }

            // Otherwise, subdivide and then add the point to whichever node will accept it
            if (sqdist(node->getPos(), n->getPos()) < OcTree::param.min_halfleng_sqr){
                return false;
            }

            Subdivide();
            for (auto const &child: children_map)
                if (child.second->Insert(node)){
                    deleteNode(); 
                    break;
                }
                       
        }
    }

    bool inserted = false;
    for (auto const &child: children_map)
        if (child.second->Insert(n)){
            inserted = true;
            break;            
        }

    return inserted;
}

bool OcTree::Insert(std::shared_ptr<Node3> n, std::unordered_set<OcTree*>& octs){
    // Ignore objects that do not belong in this oct tree
    if (!boundary.containsPoint(n->getPos())){
        if (getParent() == nullptr) {
            if (rootLimitReached)  return false;
            else                   return InsertToParent(n);
        }
        return false; // object cannot be added
    }

    if (maxDepthReached){
        if (IsEmpty()) {// If this is the first point in this oct tree, add the object here
            node = n;
            if (clevel)
                octs.insert(this);
            return true;
        }
        else // no more points accepted at this resolution
            return false;
    }

    if (leaf){

        if (boundary.getHalfLength() > OcTree::param.cluster_halfleng){
            Subdivide();
        }
        else{
            if (IsEmpty())
            {
                node = n;
                if (clevel)
                    octs.insert(this);
                return true;
            }

             // Otherwise, subdivide and then add the point to whichever node will accept it
            if (sqdist(node->getPos(), n->getPos()) < OcTree::param.min_halfleng_sqr){
                return false;
            }

            Subdivide();
            for (auto const &child: children_map)
                if (child.second->Insert(node, octs)){
                    deleteNode();
                    break;
                }
        }
    }

    for (auto const &child: children_map)
        if (child.second->Insert(n,octs)){
            if (clevel)
                octs.insert(this);
            return true;
        }

    return false;

}

int32_t OcTree::getNodeCount(){
    int32_t numNodes = 0;
    if (node !=nullptr)
        numNodes++;
    for (auto const &child: children_map)
        if (child.second != nullptr)
            numNodes += child.second->getNodeCount();
    return numNodes;
}

bool OcTree::IsNotNew(std::shared_ptr<Node3> n)
{
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be added
    }

    if (IsEmptyLeaf())
        return false ;

    if (!IsEmpty() && (sqdist(node->getPos(), n->getPos()) < OcTree::param.min_halfleng_sqr))
    {
        return true;
    }

    if (leaf)
        return false;

    for (auto const &child: children_map)
        if (child.second->IsNotNew(n))
            return true;

    return false;
}

bool OcTree::Remove(std::shared_ptr<Node3> n){
    // Ignore objects that do not belong in this oct tree
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be added
    }

    if (IsEmptyLeaf())
        return false ;

    if (!IsEmpty() && (sqdist(node->getPos(), n->getPos()) < EPS))
    {
        deleteNode();
        return true;
    }

    if (leaf)
        return false;

    bool res = false;
    for (auto const &child: children_map){
        res = child.second->Remove(n);
        if (res)
            break;
    }

    if (res)
    {
        bool res2 = true;
        for (auto const &child: children_map){
            res2 &= child.second->IsEmptyLeaf();
        }
        
        if (res2)
        {
            deleteChildren();
            deleteNode();
        }
    }

    return res;
}

bool OcTree::Remove(std::shared_ptr<Node3> n,std::unordered_set<OcTree*>& octs){
    // Ignore objects that do not belong in this oct tree
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be added
    }

    if (IsEmptyLeaf())
        return false ;

    if (!IsEmpty() && (sqdist(node->getPos(), n->getPos()) < EPS))
    {
        deleteNode();
        return true;
    }

    if (leaf)
        return false;

    bool res = false;
    for (auto const &child: children_map){
        res = child.second->Remove(n,octs);
        if (res)
            break;
    }

    if (res)
    {   
        bool res2 = true;
        for (auto const &child: children_map){
            res2 &= child.second->IsEmptyLeaf();
        }
        
        if (res2)
        {
            for (auto const &child: children_map){
                octs.erase(child.second);
            }
            deleteChildren();
            deleteNode();
        }
    }

    return res;
}

bool OcTree::Update(std::shared_ptr<Node3> n){
    // Ignore objects that do not belong in this oct tree
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be added
    }

    if (IsEmptyLeaf())
        return false ;

    if (!IsEmpty() && (sqdist(node->getPos(), n->getPos()) < EPS))
    {
        deleteNode();
        node = n;
        return true;
    }

    if (leaf)
        return false;

    for (auto const &child: children_map)
        if (child.second->Update(n))
            return true;

    return false;
}

bool OcTree::Update(std::shared_ptr<Node3> n, std::unordered_set<OcTree*>& octs){
    // Ignore objects that do not belong in this oct tree
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be added
    }

    if (IsEmptyLeaf())
        return false ;

    if (!IsEmpty() && (sqdist(node->getPos(), n->getPos()) < EPS))
    {
        deleteNode();
        node = n;
        if (clevel)
            octs.insert(this);
        return true;
    }

    if (leaf)
        return false;

    for (auto const &child: children_map)
        if (child.second->Update(n, octs)){
            if (clevel)
                octs.insert(this);
            return true;
        }   

    return false;
}

void OcTree::Subdivide()
{
    float l = boundary.getHalfLength()*0.5;
    Point3<float> c = boundary.getCenter();

    Point3<float> nwf_c = Point3<float>(c.x-l,c.y+l,c.z+l);
    AABB3 nwf(nwf_c,l);
    northWestFront = new OcTree(nwf,this);

    Point3<float> nef_c = Point3<float>(c.x+l,c.y+l,c.z+l);
    AABB3 nef(nef_c,l);
    northEastFront =  new OcTree(nef,this);

    Point3<float> swf_c = Point3<float>(c.x-l,c.y-l,c.z+l);
    AABB3 swf(swf_c,l);
    southWestFront  =  new OcTree(swf,this);

    Point3<float> sef_c = Point3<float>(c.x+l,c.y-l,c.z+l);
    AABB3 sef(sef_c,l);
    southEastFront  =  new OcTree(sef,this);

    Point3<float> nwb_c = Point3<float>(c.x-l,c.y+l,c.z-l);
    AABB3 nwb(nwb_c,l);
    northWestBack = new OcTree(nwb,this);

    Point3<float> neb_c = Point3<float>(c.x+l,c.y+l,c.z-l);
    AABB3 neb(neb_c,l);
    northEastBack =  new OcTree(neb,this);

    Point3<float> swb_c = Point3<float>(c.x-l,c.y-l,c.z-l);
    AABB3 swb(swb_c,l);
    southWestBack  =  new OcTree(swb,this);

    Point3<float> seb_c = Point3<float>(c.x+l,c.y-l,c.z-l);
    AABB3 seb(seb_c,l);
    southEastBack  =  new OcTree(seb,this);

    leaf = false;
    resetChildrenMap();

    return;
}

void OcTree::SubdivideExcept(OctChildType childType)
{
    float l = boundary.getHalfLength()*0.5;
    Point3<float> c = boundary.getCenter();

    if (childType != OctChildType::NWF)
    {
        Point3<float> nwf_c = Point3<float>(c.x-l,c.y+l,c.z+l);
        AABB3 nwf(nwf_c,l);
        northWestFront = new OcTree(nwf,this);
    }

    if (childType != OctChildType::NEF)
    {
        Point3<float> nef_c = Point3<float>(c.x+l,c.y+l,c.z+l);
        AABB3 nef(nef_c,l);
        northEastFront =  new OcTree(nef,this);
    }

    if (childType != OctChildType::SWF)
    {
         Point3<float> swf_c = Point3<float>(c.x-l,c.y-l,c.z+l);
        AABB3 swf(swf_c,l);
        southWestFront  =  new OcTree(swf,this);
    }

    if (childType != OctChildType::SEF)
    {
        Point3<float> sef_c = Point3<float>(c.x+l,c.y-l,c.z+l);
        AABB3 sef(sef_c,l);
        southEastFront  =  new OcTree(sef,this);
    }

    if (childType != OctChildType::NWB)
    {
        Point3<float> nwb_c = Point3<float>(c.x-l,c.y+l,c.z-l);
        AABB3 nwb(nwb_c,l);
        northWestBack = new OcTree(nwb,this);
    }

    if (childType != OctChildType::NEB)
    {
        Point3<float> neb_c = Point3<float>(c.x+l,c.y+l,c.z-l);
        AABB3 neb(neb_c,l);
        northEastBack =  new OcTree(neb,this);
    }

    if (childType != OctChildType::SWB)
    {
        Point3<float> swb_c = Point3<float>(c.x-l,c.y-l,c.z-l);
        AABB3 swb(swb_c,l);
        southWestBack  =  new OcTree(swb,this);
    }

    if (childType != OctChildType::SEB)
    {
        Point3<float> seb_c = Point3<float>(c.x+l,c.y-l,c.z-l);
        AABB3 seb(seb_c,l);
        southEastBack  =  new OcTree(seb,this);
    }

    leaf = false;
    resetChildrenMap();
}

 // Find all points that appear within a range
void OcTree::QueryRange(AABB3 range, std::vector<std::shared_ptr<Node3> >& nodes)
{
    // Automatically abort if the range does not intersect this oct
    if (!boundary.intersectsAABB(range) || IsEmptyLeaf()){
        return; // empty list
    }

    // Check objects at this oct level
    if (leaf){
        if (sqdist(node->getPos(), range.getCenter()) <  range.getHalfLengthSq()){
            nodes.push_back(node);
        }
        return;
    }

    // Otherwise, add the points from the children
    for (auto const &child: children_map)
        child.second->QueryRange(range,nodes);

    return ;
 }

void OcTree::getAllChildrenNonEmptyNodes(std::vector<std::shared_ptr<Node3> >& nodes)
{
     if (IsEmptyLeaf())
         return;

     if (leaf)
     {
        if (node !=nullptr){
            nodes.push_back(node);
            return;
        }
     }

     for (auto const &child: children_map)
        child.second->getAllChildrenNonEmptyNodes(nodes);

     return;
}

void OcTree::getChildNonEmptyNodes(OctChildType c, std::vector<std::shared_ptr<Node3> >& nodes)
{
     if (IsEmptyLeaf())
         return;

     if (leaf)
     {
         nodes.push_back(node);
         return;
     }

    if (children_map[c]!=nullptr)
        children_map[c]->getAllChildrenNonEmptyNodes(nodes);

    return;
}

void OcTree::QueryNonEmptyLevelC(AABB3 range, std::vector<OcTree*>& octs)
{
    // Automatically abort if the range does not intersect this oct
    if (!boundary.intersectsAABB(range) || IsEmptyLeaf()){
        return; // empty list
    }

    if (leaf){ // no children
        if (boundary.getHalfLength() > (OcTree::param.cluster_halfleng+0.0001)){
            return;
        }
    }

    if (boundary.getHalfLength() > (OcTree::param.cluster_halfleng+0.001)){
        // Otherwise, add the points from the children
        for (auto const &child: children_map)
            child.second->QueryNonEmptyLevelC(range,octs);
    }
    else
    {
        octs.push_back(this);
    }

    return ;
}

void OcTree::QueryNonEmptyLevelC(AABB3 range, std::vector<OcTree*>& octs, std::vector<float>& sqdst)
{

    // Automatically abort if the range does not intersect this oct
    if (!boundary.intersectsAABB(range) || IsEmptyLeaf()){
        return; // empty list
    }

    if (leaf){ // no children
        if (boundary.getHalfLength() > OcTree::param.cluster_halfleng+0.001){
            return;
        }
    }

    if (boundary.getHalfLength() > OcTree::param.cluster_halfleng+0.001){
        // Otherwise, add the points from the children
        for (auto const &child: children_map)
            child.second->QueryNonEmptyLevelC(range,octs,sqdst);
    }
    else
    {
        sqdst.push_back(sqdist(getCenter(),range.getCenter()));
        octs.push_back(this);
    }

    return ;
}
