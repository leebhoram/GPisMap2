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

#include "quadtree.h"

#define EPS 1e-12

float sqdist(const Point<float>& pt1, const Point<float>& pt2)
{
    float dx = (pt1.x - pt2.x);
    float dy = (pt1.y - pt2.y);
    return dx*dx + dy*dy;
}

QuadTree::QuadTree(Point<float> c):
        northWest(nullptr),
        northEast(nullptr),
        southWest(nullptr),
        southEast(nullptr),
        par(nullptr),
        maxDepthReached(false),
        rootLimitReached(false),
        leaf(true),
        clevel(false),
        node(nullptr),
        gp(nullptr)
{
     boundary = AABB(c,QuadTree::param.initroot_halfleng);
     resetChildrenMap();
}

QuadTree::QuadTree(AABB _boundary, QuadTree* const p ):
        northWest(nullptr),
        northEast(nullptr),
        southWest(nullptr),
        southEast(nullptr),
        par(nullptr),
        maxDepthReached(false),
        rootLimitReached(false),
        leaf(true),
        clevel(false),
        node(nullptr),
        gp(nullptr)
{
    boundary = _boundary;
    if (boundary.getHalfLength() < QuadTree::param.min_halfleng)
        maxDepthReached = true;
    if (boundary.getHalfLength() > QuadTree::param.max_halfleng)
        rootLimitReached = true;
    if (fabs(getHalfLength()-QuadTree::param.cluster_halfleng) < EPS)
        clevel = true;
    if (p!=nullptr)
        par = p;
    resetChildrenMap();
}

QuadTree::QuadTree(AABB _boundary,  QuadTree* const ch,  QuadChildType child_type):
        northWest(nullptr),
        northEast(nullptr),
        southWest(nullptr),
        southEast(nullptr),
        par(nullptr),
        maxDepthReached(false),
        rootLimitReached(false),
        clevel(false),
        node(nullptr),
        gp(nullptr)
{
    boundary = _boundary;
    if (boundary.getHalfLength() < QuadTree::param.min_halfleng)
        maxDepthReached = true;
    if (boundary.getHalfLength() > QuadTree::param.max_halfleng)
        rootLimitReached = true;
    if (fabs(getHalfLength()-QuadTree::param.cluster_halfleng) < EPS)
        clevel = true;
    if (child_type == QuadChildType::undefined)
    {
        leaf = true;
    }
    else
    {
        leaf = false;
        SubdivideExcept(child_type);
        if (child_type == QuadChildType::NW)
            northWest = ch;
        if (child_type == QuadChildType::NE)
            northEast = ch;
        if (child_type == QuadChildType::SW)
            southWest = ch;
        if (child_type == QuadChildType::SE)
            southEast = ch;
    }
    resetChildrenMap();
}

QuadTree::~QuadTree(){
    deleteChildren();
    deleteGP();
    deleteNode();
}

void QuadTree::deleteNode(){
    if (node != nullptr){
        node.reset();
        node = nullptr;
    }
}

void QuadTree::deleteGP(){
    if (gp != nullptr){
        gp.reset();
        gp = nullptr;
    }
}

void QuadTree::deleteChildren()
{
    if (northWest) {delete northWest; northWest = nullptr;}
    if (northEast) {delete northEast; northEast = nullptr;}
    if (southWest) {delete southWest; southWest = nullptr;}
    if (southEast) {delete southEast; southEast = nullptr;}
    leaf = true;
    resetChildrenMap();
}

void QuadTree::resetChildrenMap(){
    children_map.clear();
    children_map.insert({QuadChildType::NE, northEast});
    children_map.insert({QuadChildType::NW, northWest});
    children_map.insert({QuadChildType::SE, southEast});
    children_map.insert({QuadChildType::SW, southWest});
    updateChildrenCenter();
}

void QuadTree::updateChildrenCenter(){
    children_center.clear();
    for (auto &child: children_map)
        if (child.second!=nullptr){
            children_center.insert({child.first, child.second->getCenter()});
        }
}

std::map<QuadChildType, Point<float>> const & QuadTree::getAllChildrenCenter()
{
    return children_center;
}

std::map<QuadChildType, QuadTree*> const & QuadTree::getAllChildren()
{   
    return children_map;
}


QuadTree* const QuadTree::getRoot(){
    QuadTree* p = this;
    QuadTree* p1 = p->getParent();
    while (p1!=nullptr){
        p = p1;
        p1 = p->getParent();
    }
    return p;
}

void QuadTree::InitGP(float scale_param, float noise_param)
{
    deleteGP();
    gp = std::make_shared<OnGPIS>(scale_param, noise_param);
}

void QuadTree::UpdateGP(const vecNode& samples)
{
    if (gp != nullptr)
        gp->train(samples);
}

bool QuadTree::InsertToParent(std::shared_ptr<Node> n){
    float l = getHalfLength();
    Point<float> c = getCenter();

    // Find out what type the current node is
    const Point<float> np = n->getPos();

    Point<float> par_c;
    QuadChildType childType = QuadChildType::undefined;
    if (np.x < c.x && np.y > c.y){
        childType = QuadChildType::SE;
        par_c.x = c.x - l;
        par_c.y = c.y + l;
    }
    if (np.x > c.x && np.y > c.y){
        childType = QuadChildType::SW;
        par_c.x = c.x + l;
        par_c.y = c.y + l;
    }
    if (np.x < c.x && np.y < c.y){
        childType = QuadChildType::NE;
        par_c.x = c.x - l;
        par_c.y = c.y - l;
    }
    if (np.x > c.x && np.y < c.y){
        childType = QuadChildType::NW;
        par_c.x = c.x + l;
        par_c.y = c.y - l;
    }

    AABB parbb(par_c,2.0*l);
    par = new QuadTree(parbb,this,childType);
    return par->Insert(n);
}

bool QuadTree::Insert(std::shared_ptr<Node> n){

    // Ignore objects that do not belong in this quad tree
    if (!boundary.containsPoint(n->getPos())){
        if (getParent() == nullptr) {
            if (rootLimitReached)  return false;
            else                   return InsertToParent(n);
        }
        return false; // object cannot be added
    }

    if (maxDepthReached){
        if ( IsEmpty()) {// If this is the first point in this quad tree, add the object here
            node = n;
            return true;
        }
        else // no more points accepted at this resolution
            return false;
    }

    if (leaf){

        if (boundary.getHalfLength() > QuadTree::param.cluster_halfleng){
            Subdivide();
        }
        else{
            if (IsEmpty())
            // If this is the first point in this quad tree, add the object here
            {
                node = n;
                return true;
            }

            // If the new node is too close to the old one, do not insert and return.
            if (sqdist(node->getPos(), n->getPos()) < QuadTree::param.min_halfleng_sqr){
                return false;
            }

            // Otherwise, subdivide and then add the (old) node to whichever node will accept it
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

bool QuadTree::Insert(std::shared_ptr<Node> n,  std::unordered_set<QuadTree*>& quads)
{
     // Ignore objects that do not belong in this quad tree
    if (!boundary.containsPoint(n->getPos())){
        if (getParent() == nullptr) {
            if (rootLimitReached)  return false;
            else                   return InsertToParent(n);
        }
        return false; // object cannot be added
    }

    if (maxDepthReached){
        if ( node == nullptr) {// If this is the first point in this quad tree, add the object here
            node = n;
            if (clevel)
                quads.insert(this);
            return true;
        }
        else // no more points accepted at this resolution
            return false;
    }

    if (leaf){

        if (boundary.getHalfLength() > QuadTree::param.cluster_halfleng){
            Subdivide();
        }
        else{
            if (node == nullptr)
            // If this is the first point in this quad tree, add the object here
            {
                node = n;
                if (clevel)
                    quads.insert(this);
                return true;
            }

            // If the new node is too close to the old one, do not insert and return.
            if (sqdist(node->getPos(), n->getPos()) < QuadTree::param.min_halfleng_sqr){
                return false;
            }

            // Otherwise, subdivide and then add the point to whichever node will accept it
            Subdivide();
            for (auto const &child: children_map)
                if (child.second->Insert(node, quads)){
                    deleteNode();
                    break;
                }
        }
    }

    for (auto const &child: children_map)
        if (child.second->Insert(n,quads)){
            if (clevel)
                quads.insert(this);
            return true;
        }

    return false;
}


int32_t QuadTree::getNodeCount(){
    int32_t numNodes = 0;
    if (node !=nullptr)
        numNodes++;
    for (auto const &child: children_map)
        if (child.second != nullptr)
            numNodes += child.second->getNodeCount();
    return numNodes;
}


bool QuadTree::IsNotNew(std::shared_ptr<Node> n)
{
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be added
    }

    if (IsEmptyLeaf())
        return false ;

    if (!IsEmpty() && (sqdist(node->getPos(), n->getPos()) < QuadTree::param.min_halfleng_sqr))
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

bool QuadTree::Remove(std::shared_ptr<Node> n){
    // Ignore objects that do not belong in this quad tree
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be removed
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

bool QuadTree::Remove(std::shared_ptr<Node> n,std::unordered_set<QuadTree*>& quads){
    // Ignore objects that do not belong in this quad tree
    if (!boundary.containsPoint(n->getPos())){
        return false; // object cannot be added
    }

    if (IsEmptyLeaf())
        return false ;

    if (!IsEmpty() && (sqdist(node->getPos(), n->getPos()) < EPS))
    {
        quads.erase(this);
        deleteNode();
        return true;
    }

    if (leaf)
        return false;

    bool res = false;
    for (auto const &child: children_map){
        res = child.second->Remove(n,quads);
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
                quads.erase(child.second);
            }
            deleteChildren();
            deleteNode();
        }
    }

    return res;
}

bool QuadTree::Update(std::shared_ptr<Node> n){
    // Ignore objects that do not belong in this quad tree
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

bool QuadTree::Update(std::shared_ptr<Node> n, std::unordered_set<QuadTree*>& quads){
    // Ignore objects that do not belong in this quad tree
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
            quads.insert(this);
        return true;
    }

    if (leaf)
        return false;

    for (auto const &child: children_map)
        if (child.second->Update(n, quads)){
            if (clevel)
                quads.insert(this);
            return true;
        }

    return false;
}

void QuadTree::Subdivide()
{
    float l = boundary.getHalfLength()*0.5;
    Point<float> c = boundary.getCenter();
    Point<float> nw_c = Point<float>(c.x-l,c.y+l);
    AABB nw(nw_c,l);
    northWest = new QuadTree(nw,this);

    Point<float> ne_c = Point<float>(c.x+l,c.y+l);
    AABB ne(ne_c,l);
    northEast =  new QuadTree(ne,this);

    Point<float> sw_c = Point<float>(c.x-l,c.y-l);
    AABB sw(sw_c,l);
    southWest =  new QuadTree(sw,this);

    Point<float> se_c = Point<float>(c.x+l,c.y-l);
    AABB se(se_c,l);
    southEast =  new QuadTree(se,this);
    resetChildrenMap();

    leaf = false;

    return;
}

void QuadTree::SubdivideExcept(QuadChildType childType)
{
    float l = boundary.getHalfLength()*0.5;
    Point<float> c = boundary.getCenter();

    if (childType != QuadChildType::NW)
    {
        Point<float> nw_c = Point<float>(c.x-l,c.y+l);
        AABB nw(nw_c,l);
        northWest = new QuadTree(nw,this);
    }

    if (childType != QuadChildType::NE)
    {
        Point<float> ne_c = Point<float>(c.x+l,c.y+l);
        AABB ne(ne_c,l);
        northEast = new QuadTree(ne,this);
    }

    if (childType != QuadChildType::SW)
    {
        Point<float> sw_c = Point<float>(c.x-l,c.y-l);
        AABB sw(sw_c,l);
        southWest = new QuadTree(sw,this);
    }

    if (childType != QuadChildType::SE)
    {
        Point<float> se_c = Point<float>(c.x+l,c.y-l);
        AABB se(se_c,l);
        southEast = new QuadTree(se,this);
    }

    leaf = false;
    resetChildrenMap();
}

 // Find all points that appear within a range
void QuadTree::QueryRange(AABB range, std::vector<std::shared_ptr<Node> >& nodes)
{
    // Automatically abort if the range does not intersect this quad
    if (!boundary.intersectsAABB(range) || IsEmptyLeaf()){
        return; // empty list
    }

    // Check objects at this quad level
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

void QuadTree::getAllChildrenNonEmptyNodes(std::vector<std::shared_ptr<Node> >& nodes)
{
     if (IsEmptyLeaf())
         return;

     if (leaf)
     {
         nodes.push_back(node);
         return;
     }

    for (auto const &child: children_map)
        child.second->getAllChildrenNonEmptyNodes(nodes);

    return;
}

void QuadTree::getChildNonEmptyNodes(QuadChildType c, std::vector<std::shared_ptr<Node> >& nodes)
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


void QuadTree::QueryNonEmptyLevelC(AABB range, std::vector<QuadTree*>& quads)
{
    // Automatically abort if the range does not intersect this quad
    if (!boundary.intersectsAABB(range) || IsEmptyLeaf()){
        return; // empty list
    }

    if (leaf){ // no children
        if (boundary.getHalfLength() > (QuadTree::param.cluster_halfleng+0.0001)){
            return;
        }
    }

    if (boundary.getHalfLength() > (QuadTree::param.cluster_halfleng+0.001)){
        // Otherwise, add the points from the children
        for (auto const &child: children_map)
            child.second->QueryNonEmptyLevelC(range,quads);
    }
    else
    {
        quads.push_back(this);
    }

    return ;
}

void QuadTree::QueryNonEmptyLevelC(AABB range, std::vector<QuadTree*>& quads, std::vector<float>& sqdst)
{

    // Automatically abort if the range does not intersect this quad
    if (!boundary.intersectsAABB(range) || IsEmptyLeaf()){
        return; // empty list
    }

    if (leaf){ // no children
        if (boundary.getHalfLength() > QuadTree::param.cluster_halfleng+0.001){
            return;
        }
    }

    if (boundary.getHalfLength() > QuadTree::param.cluster_halfleng+0.001){
        // Otherwise, add the points from the children
        for (auto const &child: children_map)
            child.second->QueryNonEmptyLevelC(range,quads,sqdst);
    }
    else
    {
        sqdst.push_back(sqdist(getCenter(),range.getCenter()));
        quads.push_back(this);
    }

    return ;
}
