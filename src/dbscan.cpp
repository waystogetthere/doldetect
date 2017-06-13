#include <map>
#include <iostream>
#include <cstdio>
#include "dbscan.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

//////////////// Union set ////////////////
static vector<int> fa_, rank_;

static void init_set(int n)
{
	fa_.resize(n);
	rank_.resize(n);
	for (int i = 0; i < n; ++i)
		fa_[i] = i;
}

static int find(int x)
{
	int k, j, r;
	r = x;
	while (r != fa_[r])
		r = fa_[r];
	k = x;
	while (k != r) {
		j = fa_[k];
		fa_[k] = r;
		k = j;
	}
	return r;
}

static void union_set(int x, int y)
{
	if ((x = find(x)) == (y = find(y)))
		return;
	if (rank_[x] > rank_[y])
		fa_[y] = x;
	else {
		fa_[x] = y;
		if (rank_[x] == rank_[y])
			++rank_[y];
	}
}

void dbscan(vector<db_point>& dataset,
	vector<int>& clusters, float eps, int minpts, int datatype) {
	clusters.resize(0);
	int len = dataset.size();
	if (!len) {
		clusters.push_back(0);
		return;
	}
	// step 1: reachable points
	for (int i = 0; i < len; ++i) {
		for (int j = i + 1; j < len; ++j) {
			double dist = (datatype == 1) ? DIST_L2(dataset[i], dataset[j]) :
				DIST_L2_MAT(dataset[i].feature, dataset[j].feature);
			if (dist < eps) {
				++dataset[i].pts;
				++dataset[j].pts;
			}
		}
	}

	// step 2: core points
	vector<int> core_points;
	for (int i = 0; i < len; ++i) {
		if (dataset[i].pts >= minpts) {
			core_points.push_back(i);
			dataset[i].type = db_point::CORE;
		}
	}
	int core_num = core_points.size();
	init_set(core_num);

	// step 3: union
	/* rank_ each core point with its degree */
	for (int i = 0; i < core_num; ++i)
		rank_[i] = dataset[core_points[i]].pts;
	/* join core */
	for (int i = 0; i < core_num; ++i) {
		for (int j = i + 1; j < core_num; ++j) {
			double dist = (datatype == 1) ? DIST_L2(dataset[core_points[i]], dataset[core_points[j]]) :
				DIST_L2_MAT(dataset[core_points[i]].feature, dataset[core_points[j]].feature);
			if (dist < eps) {
				union_set(i, j);
			}
		}
	}
	/* update cluster id of core points */
	for (int i = 0; i < core_num; ++i) {
		dataset[core_points[i]].cluster_id = dataset[core_points[find(i)]].cluster_id;
	}
	/* update cluster id of border points */
	for (int i = 0; i < len; ++i)
	if (dataset[i].type != db_point::CORE)
	for (int j = 0; j < core_num; ++j) {
		double dist = (datatype == 1) ? DIST_L2(dataset[i], dataset[core_points[j]]) :
			DIST_L2_MAT(dataset[i].feature, dataset[core_points[j]].feature);
		if (dist < eps) {
			dataset[i].type = db_point::BORDER;
			dataset[i].cluster_id = dataset[core_points[fa_[j]]].cluster_id;
			break;
		}
	}

	// step 4: generate clustering result
	int cnt = 0;
	map<int, vector<int> > mmap;
	for (int i = 0; i < len; ++i)
	if (dataset[i].type != db_point::NOISE) {
		int cid = dataset[i].cluster_id;
		if (mmap.find(cid) == mmap.end()) {
			vector<int> tmp_vec;
			tmp_vec.push_back(cid);  // place the cluster center at first
			if (i != cid)
				tmp_vec.push_back(i);
			mmap[cid] = tmp_vec;
			++cnt;
		}
		else {
			if (i != cid)
				mmap[cid].push_back(i);
		}
	}
	clusters.push_back(cnt);
	for (map<int, vector<int> >::iterator it = mmap.begin(); it != mmap.end(); ++it) {
		vector<int>& tmp_vec = it->second;
		clusters.push_back(tmp_vec.size());
		for (size_t i = 0; i < tmp_vec.size(); ++i)
			clusters.push_back(tmp_vec[i]);
	}
}
