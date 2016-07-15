/*
 * ctl.h
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#ifndef CTL_H
#define CTL_H

#include "stdio.h"
#include "utils.h"
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <set>

using namespace std;

struct Document {
  double* delta;
  double* u;
  double* log_gamma;
  int* labels_ptr;
  int* words_ptr;
  int* words_cnt_ptr;
  double alpha, beta;
  int num_labels;
  int num_words;
  int num_topics;
  int num_all_labels;
  int num_all_words;
  double* topic;
  double lik;
  Document(int* labels_ptr_,int* words_ptr_,int* words_cnt_ptr_,int num_labels_,int num_words_,int num_topics_) {
    num_topics = num_topics_;
    num_labels = num_labels_;
    num_words = num_words_;
    log_gamma = new double[num_words * num_topics];
    topic = new double[num_topics];
    labels_ptr = labels_ptr_;
    words_ptr = words_ptr_;
    words_cnt_ptr = words_cnt_ptr_;
    lik = 100;
    init();
  }
  ~Document() {
    if (log_gamma) delete[] log_gamma;
    if (labels_ptr) delete[] labels_ptr;
    if (words_ptr) delete[] words_ptr;
    if (words_cnt_ptr) delete[] words_cnt_ptr;
    if (topic) delete[] topic;
    if (u) delete[] u;
    if (delta) delete[] delta;
  }
  void init_delta_u(const int& all_num_labels_);
  void init();
  Document* convert_to_unlabel(int all_num_labels) {
    int* labels_ptr_ = new int [all_num_labels];
    int* words_ptr_ = new int [num_words];
    int* words_cnt_ptr_ = new int [num_words];
    memcpy(words_ptr_,words_ptr, sizeof(int) * num_words);
    memcpy(words_cnt_ptr_,words_cnt_ptr,sizeof(int)*num_words);
    for (int i = 0; i < all_num_labels; i++) labels_ptr_[i] = i;
    return new Document(labels_ptr_,words_ptr_,words_cnt_ptr_,all_num_labels,num_words,num_topics);
  }

};

struct ctl_model {
  int num_docs;
  int num_words;
  int num_topics;
  int num_labels;
  int num_all_words;
  double* sigma;
  double* inv_sigma;
  double* mu;
  double* Lambda;
  double* lambda;
  double* log_phi;
  ctl_model(int num_docs_,int num_words_,int num_topics_,int num_labels_, int num_all_words_, ctl_model* init_model=NULL) {
    num_labels = num_labels_;
    num_docs = num_docs_;
    num_topics = num_topics_;
    num_words = num_words_;
    num_all_words = num_all_words_;
    sigma = new double[num_labels * num_labels];
    inv_sigma = new double[num_labels * num_labels];
    mu = new double[num_labels];
    Lambda = new double[num_topics];
    lambda = new double[num_labels * num_topics];
    log_phi = new double[num_topics * num_words];
  }
  ctl_model(char* model_root, char* prefix) {
    read_model_info(model_root);
    char sigma_file[1000];
    char inv_sigma_file[1000];
    char mu_file[1000];
    char Lambda_file[1000];
    char phi_file[1000];
    char lambda_file[1000];

    sprintf(sigma_file, "%s/%s.sigma", model_root, prefix);
    sprintf(inv_sigma_file, "%s/%s.inv_sigma", model_root, prefix);
    sprintf(mu_file, "%s/%s.mu", model_root, prefix);
    sprintf(Lambda_file, "%s/%s.Lambda", model_root, prefix);
    sprintf(phi_file, "%s/%s.phi", model_root, prefix);
    sprintf(lambda_file,"%s/%s.lambda", model_root, prefix);

    sigma = load_mat(sigma_file, num_labels, num_labels);
    inv_sigma = load_mat(inv_sigma_file, num_labels, num_labels);
    mu = load_mat(mu_file, 1, num_labels);
    Lambda = load_mat(Lambda_file, 1, num_topics);
    log_phi = load_mat(phi_file, num_topics, num_words);
    lambda = load_mat(lambda_file, num_labels, num_topics);
  }
  ~ctl_model() {
    if (mu)delete[] mu;
    if (sigma) delete[] sigma;
    if (inv_sigma) delete[] inv_sigma;
    if (log_phi) delete[] log_phi;
    if (Lambda) delete[] Lambda;
    if (lambda) delete[] lambda;
  }
  double* load_mat(char* filename,int row,int col);
  void read_model_info(char* model_root);
  void set_model(ctl_model* model);
};

struct onlineModel {
  int onlinenum_docs;
  int onlinenum_words;
  int onlinenum_topics;
  int onlinenum_labels;
  double* onlinesigma;
  double* onlineinv_sigma;
  double* onlinemu;
  double* onlineLambda;
  double* onlinelambda;
  double* onlinelog_phi;
  onlineModel(int num_docs_,int num_words_,int num_topics_,int num_labels_, ctl_model* init_model=NULL) {
	  onlinenum_labels = num_labels_;
	  onlinenum_docs = num_docs_;
	  onlinenum_topics = num_topics_;
	  onlinenum_words = num_words_;
    onlinesigma = new double[onlinenum_labels * onlinenum_labels];
    onlineinv_sigma = new double[onlinenum_labels * onlinenum_labels];
    onlinemu = new double[onlinenum_labels];
    onlineLambda = new double[onlinenum_topics];
    onlinelambda = new double[onlinenum_labels * onlinenum_topics];
    onlinelog_phi = new double[onlinenum_topics * onlinenum_words];
  }
  onlineModel(char* model_root, char* prefix) {
	  onlineCTLread_model_info(model_root);
    char onlinesigma_file[1000];
    char onlineinv_sigma_file[1000];
    char onlinemu_file[1000];
    char onlineLambda_file[1000];
    char onlinephi_file[1000];
    char onlinelambda_file[1000];
    char onlineu_file[1000];

    sprintf(onlinesigma_file, "%s/%s.sigma", model_root, prefix);
    sprintf(onlineinv_sigma_file, "%s/%s.inv_sigma", model_root, prefix);
    sprintf(onlinemu_file, "%s/%s.mu", model_root, prefix);
    sprintf(onlineLambda_file, "%s/%s.Lambda", model_root, prefix);
    sprintf(onlinephi_file, "%s/%s.phi", model_root, prefix);
    sprintf(onlinelambda_file,"%s/%s.lambda", model_root, prefix);

    onlinesigma = onlineCTLload_mat(onlinesigma_file, onlinenum_labels, onlinenum_labels);
    onlineinv_sigma = onlineCTLload_mat(onlineinv_sigma_file, onlinenum_labels, onlinenum_labels);
    onlinemu = onlineCTLload_mat(onlinemu_file, 1, onlinenum_labels);
    onlineLambda = onlineCTLload_mat(onlineLambda_file, 1, onlinenum_topics);
    onlinelog_phi = onlineCTLload_mat(onlinephi_file, onlinenum_topics, onlinenum_words);
    onlinelambda = onlineCTLload_mat(onlinelambda_file, onlinenum_labels, onlinenum_topics);
  }
  ~onlineModel() {
    if (onlinemu)delete[] onlinemu;
    if (onlinesigma) delete[] onlinesigma;
    if (onlineinv_sigma) delete[] onlineinv_sigma;
    if (onlinelog_phi) delete[] onlinelog_phi;
    if (onlineLambda) delete[] onlineLambda;
    if (onlinelambda) delete[] onlinelambda;
  }
  double* onlineCTLload_mat(char* filename,int row,int col);
  void onlineCTLread_model_info(char* model_root);
  void onlineCTLset_model(onlineModel* model);
};

struct Config {
  double pi_learn_rate;
  int max_pi_iter;
  double pi_min_eps;
  double xi_learn_rate;
  int max_xi_iter;
  double xi_min_eps;
  int max_em_iter;
  static bool print_debuginfo;
  int num_threads;
  int max_var_iter;
  double var_converence;
  double em_converence;
  int level;
  double lambda_learn_rate;
  double lambda_max_iter;
  double lambda_min_eps;
  double u_learn_rate;
  double u_max_iter;
  double u_min_eps;
  int total_num_words;
  int total_num_tags;
  Config(char* settingfile) {
    pi_learn_rate = 0.00001;
    max_pi_iter = 100;
    pi_min_eps = 1e-5;
    max_xi_iter = 100;
    xi_learn_rate = 10;
    xi_min_eps = 1e-5;
    max_em_iter = 30;
    num_threads = 1;
    var_converence = 1e-6;
    max_var_iter = 30;
    em_converence = 1e-4;
    level = 20;
    lambda_learn_rate = 1e-4;
    lambda_max_iter = 30;
    lambda_min_eps = 1e-4;
    u_learn_rate = 1e-2;
    u_max_iter = 20;
    u_min_eps = 1e-4;
    total_num_words = 0;
    total_num_tags = 0;
    if(settingfile) read_settingfile(settingfile);
  }
  void read_settingfile(char* settingfile);
};

void begin_ctl(char* inputfile, Config config, int num_topics, int total_num_words,char* model_root,
		int total_num_labels,onlineModel* onlinemodel, double rho_b,set<int> &existedTags,set<int> &newTags,char* batchdatafile);

#endif

