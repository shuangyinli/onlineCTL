/*
 * ctl.cc
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#include "onlinectl.h"

#include "utils.h"
#include "params.h"
#include <set>
#include <assert.h>
#include "onlinectl-estimate.h"
#include "onlinectl-inference.h"
#include "onlinectl-learn.h"
#include "onlinectl-ss.h"

pthread_mutex_t mutexx = PTHREAD_MUTEX_INITIALIZER;

bool Config::print_debuginfo = true;

void Document::init_delta_u(const int &num_all_labels_) {
  num_all_labels = num_all_labels_;
  delta = new double[num_all_labels];
  u = new double[num_all_labels];
  memset(delta, 0, sizeof(double) * num_all_labels);
  for (int i = 0; i < num_all_labels; ++ i) u[i] = 0;
}

Document** read_batchdata(char* filename,int num_topics, int& num_docs, int& batch_num_all_words,int num_labels,set<int> &existedTags,set<int> &newTags, set<int> &Tagb_) {
  num_docs = 0;
  batch_num_all_words = 0;
  FILE* fp = fopen(filename,"r"); //calculate the file line num
  char c;
  while((c=getc(fp))!=EOF) {
    if (c=='\n') num_docs++;
  }
  fclose(fp);
  fp = fopen(filename,"r");
  int doc_num_labels;
  int doc_num_words;
  char str[10];
  Document** corpus = new Document* [num_docs + 10];
  num_docs = 0;
  while(fscanf(fp,"%d",&doc_num_labels) != EOF) {
    int* labels_ptr = new int[doc_num_labels];
    for (int i = 0; i < doc_num_labels; i++) {
      fscanf(fp,"%d",&labels_ptr[i]);
      if(existedTags.count(labels_ptr[i]) ==0){ //
    	  newTags.insert(labels_ptr[i]);
      }else{
    	  Tagb_.insert(labels_ptr[i]);
      }
    }
    fscanf(fp,"%s",str); //read @
    fscanf(fp,"%d", &doc_num_words);
    int* words_ptr = new int[doc_num_words];
    int* words_cnt_ptr = new int [doc_num_words];
    for (int i =0; i < doc_num_words; i++) {
      fscanf(fp,"%d:%d", &words_ptr[i],&words_cnt_ptr[i]);
      batch_num_all_words+= words_cnt_ptr[i];
    }
    corpus[num_docs++] = new Document(labels_ptr, words_ptr, words_cnt_ptr, doc_num_labels, doc_num_words, num_topics);
  }
  fclose(fp);

  for (int i = 0; i < num_docs; ++ i) corpus[i]->init_delta_u(num_labels);
  return corpus;
}

/*void read_Totaldata(char* filename,int& num_words, int& num_docs, int& num_labels, int& num_all_words) {
  num_words = 0;
  num_docs = 0;
  num_labels = 0;
  num_all_words = 0;
  FILE* fp = fopen(filename,"r"); //calcaulte the file line num
  char c;
  while((c=getc(fp))!=EOF) {
    if (c=='\n') num_docs++;
  }
  fclose(fp);
  fp = fopen(filename,"r");
  int doc_num_labels;
  int doc_num_words;
  char str[10];
  num_docs = 0;
  while(fscanf(fp,"%d",&doc_num_labels) != EOF) {
    int* labels_ptr = new int[doc_num_labels];
    for (int i = 0; i < doc_num_labels; i++) {
      fscanf(fp,"%d",&labels_ptr[i]);
      num_labels = num_labels > labels_ptr[i]?num_labels:labels_ptr[i];
    }

    fscanf(fp,"%s",str);
    fscanf(fp,"%d", &doc_num_words);
    int* words_ptr = new int[doc_num_words];
    int* words_cnt_ptr = new int [doc_num_words];
    for (int i =0; i < doc_num_words; i++) {
      fscanf(fp,"%d:%d", &words_ptr[i],&words_cnt_ptr[i]);
      num_words = num_words < words_ptr[i]?words_ptr[i]:num_words;
      num_all_words += words_cnt_ptr[i];
    }
    //corpus[num_docs++] = new Document(labels_ptr, words_ptr, words_cnt_ptr, doc_num_labels, doc_num_words, num_topics);
    delete[] labels_ptr;
    delete[] words_ptr;
    delete[] words_cnt_ptr;
  }
  fclose(fp);
  num_words ++;
  num_labels ++;

  printf("num_docs: %d\nnum_labels: %d\nnum_words:%d\n",num_docs,num_labels,num_words);
}*/

void Config::read_settingfile(char* settingfile) {
  FILE* fp = fopen(settingfile,"r");
  char key[100];
  while (fscanf(fp,"%s",key)!=EOF){
    if (strcmp(key,"pi_learn_rate")==0) {
      fscanf(fp,"%lf",&pi_learn_rate);
      continue;
    }
    if (strcmp(key,"max_pi_iter") == 0) {
      fscanf(fp,"%d",&max_pi_iter);
      continue;
    }
    if (strcmp(key,"pi_min_eps") == 0) {
      fscanf(fp,"%lf",&pi_min_eps);
      continue;
    }
    if (strcmp(key,"xi_learn_rate") == 0) {
      fscanf(fp,"%lf",&xi_learn_rate);
      continue;
    }
    if (strcmp(key,"max_xi_iter") == 0) {
      fscanf(fp,"%d",&max_xi_iter);
      continue;
    }
    if (strcmp(key,"xi_min_eps") == 0) {
      fscanf(fp,"%lf",&xi_min_eps);
      continue;
    }
    if (strcmp(key,"max_em_iter") == 0) {
      fscanf(fp,"%d",&max_em_iter);
      continue;
    }
    if (strcmp(key,"num_threads") == 0) {
      fscanf(fp, "%d", &num_threads);
    }
    if (strcmp(key, "var_converence") == 0) {
      fscanf(fp, "%lf", &var_converence);
    }
    if (strcmp(key, "max_var_iter") == 0) {
      fscanf(fp, "%d", &max_var_iter);
    }
    if (strcmp(key, "em_converence") == 0) {
      fscanf(fp, "%lf", &em_converence);
    }
    if (strcmp(key, "level") == 0) {
      fscanf(fp, "%d", &level);
    }
    if (strcmp(key, "total_num_words") == 0) {
      fscanf(fp, "%d", &total_num_words);
    }

    if (strcmp(key, "total_num_tags") == 0) {
      fscanf(fp, "%d", &total_num_tags);
    }

  }
}

void Document::init() {
  num_all_words = 0;
  for (int i = 0; i < num_words; i++) {
    num_all_words += words_cnt_ptr[i];
    double sum_log_gamma = 0.0;
    for (int k = 0; k < num_topics; k++) {
      log_gamma[i * num_topics + k] = util::random();
      sum_log_gamma += log_gamma[i * num_topics + k]; 
    }
    for (int k = 0; k < num_topics; k++) {
      log_gamma[i * num_topics + k] /= sum_log_gamma;
      log_gamma[i * num_topics + k] = log(log_gamma[i * num_topics + k]);
    }
  }
}
void print_mat(double* mat, int row, int col, char* filename) {
  FILE* fp = fopen(filename,"w");
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      fprintf(fp,"%lf ",mat[i*col + j]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

void print_documents_u(Document** corpus, ctl_model* model, char* output_dir) {
  char filename[1000];
  int num_docs = model->num_docs;
  sprintf(filename, "%s/doc-u.txt", output_dir);
  FILE* fp = fopen(filename,"w");
  for (int i = 0; i < num_docs; ++ i) {
    Document* doc = corpus[i];
    for (int k = 0; k < model->num_labels; ++ k) fprintf(fp, "%lf ", doc->u[k]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void print_documents_topics(Document** corpus, int num_docs, char* output_dir) {
  char filename[1000];
  sprintf(filename, "%s/doc-topics-dis.txt", output_dir);
  char liks_file[1000];
  sprintf(liks_file, "%s/likehoods.txt", output_dir);
  FILE* liks_fp = fopen(liks_file, "w");
  FILE* fp = fopen(filename,"w");
  for (int i = 0; i < num_docs; i++) {
    Document* doc = corpus[i];
    fprintf(fp, "%lf", doc->topic[0]);
    fprintf(liks_fp, "%lf\n", doc->lik);
    for (int k = 1; k < doc->num_topics; k++) fprintf(fp, " %lf", doc->topic[k]);
    fprintf(fp, "\n");
  }
  fclose(fp);
  fclose(liks_fp);
}

void print_onlineCTLmodel(char* model_root, onlineModel* onlinemodel,char* batchdatafile) {
	char sigma_file[1000];
	char inv_sigma_file[1000];
	char mu_file[1000];
	char Lambda_file[1000];
	char phi_file[1000];
	char lambda_file[1000];
	sprintf(sigma_file, "%s/onlinectl_%s.sigma", model_root,batchdatafile);
	sprintf(inv_sigma_file, "%s/onlinectl_%s.inv_sigma", model_root,batchdatafile);
	sprintf(mu_file, "%s/onlinectl_%s.mu", model_root,batchdatafile);
	sprintf(Lambda_file, "%s/onlinectl_%s.Lambda", model_root,batchdatafile);
	sprintf(phi_file, "%s/onlinectl_%s.phi", model_root,batchdatafile);
	sprintf(lambda_file, "%s/onlinectl_%s.lambda", model_root,batchdatafile);

	print_mat(onlinemodel->onlinesigma, onlinemodel->onlinenum_labels,
			onlinemodel->onlinenum_labels, sigma_file);
	print_mat(onlinemodel->onlineinv_sigma, onlinemodel->onlinenum_labels,
			onlinemodel->onlinenum_labels, inv_sigma_file);
	print_mat(onlinemodel->onlinemu, 1, onlinemodel->onlinenum_labels, mu_file);
	print_mat(onlinemodel->onlineLambda, 1, onlinemodel->onlinenum_topics,
			Lambda_file);
	print_mat(onlinemodel->onlinelog_phi, onlinemodel->onlinenum_topics,
			onlinemodel->onlinenum_words, phi_file);
	print_mat(onlinemodel->onlinelambda, onlinemodel->onlinenum_labels,
			onlinemodel->onlinenum_topics, lambda_file);
}
void print_batchdata_para(Document** corpus, char* model_root, ctl_model* model, char* batchname) {
	  char topic_dis_file[1000];
	  char u_file[1000];
	  sprintf(topic_dis_file,"%s/%s_final.topic_dis", model_root, batchname);
	  sprintf(u_file,"%s/%s_final.u", model_root, batchname);

	  FILE* topic_dis_fp = fopen(topic_dis_file,"w");
	   FILE* u_fp = fopen(u_file, "w");
	   int num_docs = model->num_docs;
	   for (int d = 0; d < num_docs; d++) {
	     Document* doc = corpus[d];
	     for (int k = 0; k < doc->num_topics; k++)fprintf(topic_dis_fp, "%lf ", doc->topic[k]);
	     for (int k = 0; k < model->num_labels; ++ k) fprintf(u_fp, "%lf ", doc->u[k]);
	     fprintf(topic_dis_fp, "\n");
	     fprintf(u_fp, "\n");
	   }
	   fclose(u_fp);
	   fclose(topic_dis_fp);
}

void ctl_model::set_model(ctl_model* model) {
  memcpy(sigma, model->sigma, sizeof(double) * num_labels * num_labels);
  memcpy(inv_sigma, model->inv_sigma, sizeof(double) * num_labels * num_labels);
  memcpy(mu, model->mu, sizeof(double) * num_labels);
  memcpy(Lambda, model->Lambda, sizeof(double) * num_topics);
  memcpy(lambda, model->lambda, sizeof(double) * num_labels * num_topics);
  memcpy(log_phi, model->log_phi, sizeof(double) * num_topics * num_words);
}

void ctl_model::read_model_info(char* model_root) {
  char filename[1000];
  sprintf(filename, "%s/model.info",model_root);
  printf("%s\n",filename);
  FILE* fp = fopen(filename,"r");
  char str[100];
  int value;
  while (fscanf(fp,"%s%d",str,&value)!=EOF) {
    if (strcmp(str,"num_labels:") == 0)num_labels = value;
    if (strcmp(str, "num_words:") == 0)num_words = value;
    if (strcmp(str, "num_topics:") == 0)num_topics = value;
  }
  printf("num_labels: %d\nnum_words: %d\nnum_topics: %d\n",num_labels,num_words, num_topics);
  fclose(fp);
}

double* ctl_model::load_mat(char* filename, int row, int col) {
  FILE* fp = fopen(filename,"r");
  double* mat = new double[row * col];
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      fscanf(fp, "%lf", &mat[i*col+j]);
    }
  }
  fclose(fp);
  return mat;
}

double* onlineModel::onlineCTLload_mat(char* filename, int row, int col) {
  FILE* fp = fopen(filename,"r");
  double* mat = new double[row * col];
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      fscanf(fp, "%lf", &mat[i*col+j]);
    }
  }
  fclose(fp);
  return mat;
}

void onlineModel::onlineCTLread_model_info(char* model_root) {
  char filename[1000];
  sprintf(filename, "%s/model.info",model_root);
  printf("%s\n",filename);
  FILE* fp = fopen(filename,"r");
  char str[100];
  int value;
  while (fscanf(fp,"%s%d",str,&value)!=EOF) {
    if (strcmp(str,"num_labels:") == 0)onlinenum_labels = value;
    if (strcmp(str, "num_words:") == 0)onlinenum_words = value;
    if (strcmp(str, "num_topics:") == 0)onlinenum_topics = value;
  }
  printf("num_labels: %d\nnum_words: %d\nnum_topics: %d\n",onlinenum_labels,onlinenum_words, onlinenum_topics);
  fclose(fp);
}

void onlineModel::onlineCTLset_model(onlineModel* model) {
  memcpy(onlinesigma, model->onlinesigma, sizeof(double) * onlinenum_labels * onlinenum_labels);
  memcpy(onlineinv_sigma, model->onlineinv_sigma, sizeof(double) * onlinenum_labels * onlinenum_labels);
  memcpy(onlinemu, model->onlinemu, sizeof(double) * onlinenum_labels);
  memcpy(onlineLambda, model->onlineLambda, sizeof(double) * onlinenum_topics);
  memcpy(onlinelambda, model->onlinelambda, sizeof(double) * onlinenum_labels * onlinenum_topics);
  memcpy(onlinelog_phi, model->onlinelog_phi, sizeof(double) * onlinenum_topics * onlinenum_words);
}

char** getFileNameArray(const char *path, int & countfile){
		int count = 0;
		char **fileNameList = NULL;
		struct dirent* ent = NULL;
		DIR *pDir;
		char dir[512];
		struct stat statbuf;

		if ((pDir = opendir(path)) == NULL) {
			printf("Cannot open directory:%s\n", path);
			return 0;
		}

		while ((ent = readdir(pDir)) != NULL) {
			snprintf(dir, 512, "%s/%s", path, ent->d_name);
			lstat(dir, &statbuf);
			if (!S_ISDIR(statbuf.st_mode)) {
				count++;
			}
		}
		closedir(pDir);
		//printf("共%d个文件\n", count);

		if ((fileNameList = (char**) malloc(sizeof(char*) * count)) == NULL) {
			printf("Malloc heap failed!\n");
			return 0;
		}
		if ((pDir = opendir(path)) == NULL) {
			printf("cannot open the path!\n");
			return 0;
		}
		int i;
		for (i = 0; (ent = readdir(pDir)) != NULL && i < count;) {
			if (strlen(ent->d_name) <= 0)
				continue;
			snprintf(dir, 512, "%s/%s", path, ent->d_name);
			lstat(dir, &statbuf);
			if (!S_ISDIR(statbuf.st_mode)) {
				if ((fileNameList[i] = (char*) malloc(strlen(ent->d_name) + 1)) == NULL) {
					return 0;
				}
				memset(fileNameList[i], 0, strlen(ent->d_name) + 1);
				strcpy(fileNameList[i], ent->d_name);
				i++;
			}

		}
		countfile = count;
		//printf("i = %d\n",i);
		closedir(pDir);
		//printf("%d\n",sizeof(fileNameList));
		return fileNameList;
}

void begin_onlinectl(char* inputpath, char* settingfile,int num_topics, char* model_root) {
  setbuf(stdout,NULL);
  int total_num_docs=0;
  int total_num_words;
  int total_num_tags;
  srand(unsigned(time(0)));

  // read total_num_words  total_num_labels
  Config config = Config(settingfile);
  if(config.total_num_tags == 0){
	  printf("please set the total tag number in the setting.txt \n");
	  exit(0);
  }

  if(config.total_num_words == 0){
	  printf("please set the total word number in the setting.txt \n");
	  exit(0);
  }

   total_num_words = config.total_num_words;
   total_num_tags = config.total_num_tags;

  //read_Totaldata(inputtotalfile,total_num_words,total_num_docs,total_num_labels, total_num_all_words);
  default_params();

  onlineModel* onlinemodel = new onlineModel(total_num_docs, total_num_words,num_topics,total_num_tags);
  //Initialize
  init_lambda(onlinemodel->onlinelambda, onlinemodel->onlinenum_labels * onlinemodel->onlinenum_topics);

	int countfiles = 0;
	char ** fileNameList = getFileNameArray(inputpath, countfiles);
	int b;
	double kppa = 0.75;
	int tau_0 = 1024;
	int updatect = 0;
	set<int> existedTags;
	set<int> newTags;
	puts("BEGIN!!");
	for (b = 0; b < countfiles; b++) {
		char * batchdatafile = fileNameList[b];
		printf("******************%d**********************\n",b);
		printf("now begin the %d batch: %s...\n",b,batchdatafile);
		char* inputfile = (char *)malloc(strlen(inputpath)+strlen(batchdatafile) +1);
		if (inputfile == NULL) {
			printf("the input batch file is wrong, exit(0)\n");
			exit(0);
		}
		double rho_b = pow(tau_0+updatect, -kppa);
		strcpy(inputfile, inputpath);
		strcat(inputfile,batchdatafile);
		//printf("the input batch file is %s.\n", inputfile);
		begin_ctl(inputfile, config,num_topics,total_num_words,model_root,total_num_tags, onlinemodel, rho_b,existedTags,newTags,batchdatafile);
		print_onlineCTLmodel(model_root, onlinemodel,batchdatafile);
		updatect ++;
		printf("the input batch %d is over \n\n",b);
	}

}

void begin_ctl(char* inputfile, Config config,int num_topics, int total_num_words, char* model_root, int total_num_tags, onlineModel* onlinemodel, double rho_b, set<int> &existedTags, set<int> &newTags, char* batchdatafile) {
  setbuf(stdout,NULL);
  int num_docs;
  int batch_num_all_words;
  srand(unsigned(time(0)));
  set<int> Tagb_;
  Document** corpus = read_batchdata(inputfile,num_topics,num_docs,batch_num_all_words,total_num_tags,existedTags, newTags,Tagb_);
  puts("Read batch Data Finish.");
  printf("This batch contains %d documents and %d words, and adds %d new tags.\n", num_docs, batch_num_all_words, newTags.size());
  puts("Now begin to train the batch...");
  default_params();
  ctl_model* model = new ctl_model(num_docs,total_num_words,num_topics,total_num_tags,batch_num_all_words);
  ctl_model* old_model = new ctl_model(num_docs,total_num_words,num_topics,total_num_tags,batch_num_all_words);
  SS* ss = new SS(total_num_tags * total_num_tags, total_num_tags);

  time_t learn_begin_time = time(0);
  int num_round = 0;
  //printf("cal likehood...\n");
  double lik = compute_corpus_log_likehood(corpus, model);
  double lik2 = compute_corpus_log_likehood2(corpus, model);
  double old_lik;
  //printf("lik %lf\n", lik);
  double plik;
  double* likehood_record = new double [config.max_em_iter];
  likehood_record[0] = lik;
  double converged = 1;
  //Initialize
  init_lambda(model->lambda, model->num_labels * model->num_topics);
  //puts("Init Lambda Finish!!");
  inference_Lambda(model->Lambda, model->lambda, model->num_topics, model->num_labels, 0);
  init_sigma_mu_phi(model);
  //Finish Initialize
  //
  do {
    //Init SS
    old_model->set_model(model);
    old_lik = lik2;
    ss->init();
    time_t cur_round_begin_time = time(0);
    plik = lik;
    //printf("Round %d begin...\n", num_round ++);
    num_round ++;
    //printf("inference...\n");
    //E-step
    run_thread_inference(corpus, model, &config, ss);
    inference_lambda(corpus, model, &config);
    inference_Lambda(model->Lambda, model->lambda, model->num_topics, model->num_labels, 0);
    if (compute_corpus_log_likehood2(corpus, model) < old_lik) model->set_model(old_model);

    //M-step
    //printf("learn phi...\n");
    learn_phi(corpus, model);
    //printf("learn mu...\n");
    learn_mu(model, ss);
    //printf("learn sigma...\n");
    learn_sigma(model, ss);
    //printf("cal likehood...\n");
    lik = compute_corpus_log_likehood(corpus, model);
    lik2 = compute_corpus_log_likehood2(corpus, model);
    double perplex = exp(-lik2/batch_num_all_words);
    converged = (plik - lik) / plik;
    //unsigned int cur_round_cost_time = time(0) - cur_round_begin_time;
    printf("                   Round %d: likehood=%lf likehood2=%lf perplexity=%lf converge=%lf.\n",num_round,lik,lik2,perplex,converged);
    likehood_record[num_round] = lik;
  } while (num_round < config.max_em_iter);
  unsigned int learn_cost_time = time(0) - learn_begin_time;
  printf("					This batch runs %d rounds and cost %u secs.\n", num_round, learn_cost_time);
  print_batchdata_para(corpus, model_root,  model, batchdatafile);
  //print_lik(likehood_record, num_round, model_root);
  //print_para(corpus, -1, model_root, model);

//update online model
//phi
  printf("Now begin to update the model parameters...the rho_b is %f . \n",rho_b);
  puts("First, update the phi...");
	int num_words = onlinemodel->onlinenum_words;
	for (int k = 0; k < num_topics; k++) {
				for (int i = 0; i < num_words; i++) {
					onlinemodel->onlinelog_phi[k * num_words + i] =  log(
							(1 - rho_b)* exp(onlinemodel->onlinelog_phi[k * num_words+ i])
									+ rho_b * 1024 * exp(model->log_phi[k * num_words + i]) / num_docs);
				}
	}
	normalize_log_matrix_rows(onlinemodel->onlinelog_phi, num_topics, onlinemodel->onlinenum_words);

	puts("Then, update the lambdas and mu...");
	set<int>::iterator eit;
	double* delta_descent_lambda = new double[num_topics];
	for (eit = existedTags.begin(); eit != existedTags.end(); eit++) {
		int labelid = *eit;
		//lambda
		get_descent_lambda_l(corpus, model, delta_descent_lambda, labelid);
		//lambda
		for (int k = 0; k < num_topics; k++) {
			onlinemodel->onlinelambda[labelid*num_topics +k] = onlinemodel->onlinelambda[labelid*num_topics +k] - rho_b * delta_descent_lambda[k];
		}
		//mu
		onlinemodel->onlinemu[labelid] = (1 - rho_b) * onlinemodel->onlinemu[labelid] + rho_b * model->mu[labelid];
	}
	delete [] delta_descent_lambda;

	set<int>::iterator nit;
	for (nit = newTags.begin(); nit != newTags.end(); nit++) {
		int labelid = *nit;
		//lambda
		for (int k = 0; k < num_topics; k++) {
			onlinemodel->onlinelambda[labelid*num_topics +k] = model->lambda[labelid*num_topics +k];
		}
		//mu
		onlinemodel->onlinemu[labelid] = model->mu[labelid];
	}
	//sigma
	puts("Following, update the Sigma...");
	 for (int i = 0; i < total_num_tags; ++ i) {
			 if(Tagb_.count(i) != 0){
				 for (int j = 0; j < total_num_tags; ++ j) {
					 if(newTags.count(i) == 0){
						 onlinemodel->onlinesigma[i*total_num_tags +j] = (1 - rho_b) * onlinemodel->onlinesigma[i*total_num_tags +j]
																											+ rho_b * model->sigma[i*total_num_tags +j];
 					 }else if (newTags.count(i) != 0){
 						onlinemodel->onlinesigma[i*total_num_tags +j] = model->sigma[i*total_num_tags +j];
 					 }
				 }
			 }else if(newTags.count(i) != 0){
				 for (int j = 0; j < total_num_tags; ++ j)
					 onlinemodel->onlinesigma[i*total_num_tags +j] = model->sigma[i*total_num_tags +j];
			 } else{
				 for(int j = 0; j < total_num_tags; ++ j) {
					 if(Tagb_.count(j)!=0){
						 onlinemodel->onlinesigma[i*total_num_tags +j] = (1 - rho_b) * onlinemodel->onlinesigma[i*total_num_tags +j]
						 																											+ rho_b * model->sigma[i*total_num_tags +j];
					 }if(newTags.count(j) != 0){
						 onlinemodel->onlinesigma[i*total_num_tags +j] = model->sigma[i*total_num_tags +j];
					 }
				 }
		 }
	}
	 puts("Last, update the Lambda...");
 // Lambda
	for (int k = 0; k < num_topics; k++) {
		inference_Lambda(onlinemodel->onlineLambda, onlinemodel->onlinelambda, onlinemodel->onlinenum_topics, onlinemodel->onlinenum_labels, 0);
	}

//clear newtags set
  set<int>::iterator it;
  for(it=newTags.begin();it!=newTags.end();it++){
	  existedTags.insert(*it);
  }
  newTags.clear();

  for (int i = 0; i < num_docs; i++) delete corpus[i];
  delete[] likehood_record;
  delete model;
  delete[] corpus;
  delete ss;
}


int main(int argc, char* argv[]) {
  if (argc <= 1 || (!(strcmp(argv[1],"est") == 0 && argc == 6))) {
    printf("./onlinectl est <input data directory> <setting.txt> <num_topics> <model save dir>\n");
    return 1;
  }
  if (argc > 1 && strcmp(argv[1],"est") == 0) begin_onlinectl(argv[2],argv[3],atoi(argv[4]),argv[5]);
  //./onlinectl est  <input data directory> <setting.txt> <num_topics> <model save dir>
  return 0;
}

