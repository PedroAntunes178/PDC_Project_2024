#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "load.h"
#include "mpi.h"

#ifndef OMP_NUM_THREADS
#define OMP_NUM_THREADS 1
#endif

void entry(float input[1][1][28][28], float output[1][10]);
int find_max(float result[1][10]);

int main(int argc, char *argv[]) {
	clock_t pre_inference;
	clock_t pos_inference;

	int label=0;
	float input[1][1][28][28];
	float result[1][10];
	int correct=0, correct_sum=0;
	int wrong=0, wrong_sum=0;
	int correct_bins[10]={0};
	int correct_bins_sum[10]={0};
	int wrong_bins[10]={0};
	int wrong_bins_sum[10]={0};

	// MPI variables
	int n_tasks, rank, rc;
	MPI_Status status;
	// Init MPI
	rc = MPI_Init(&argc, &argv);
	if(rc!=MPI_SUCCESS){
		printf("Error starting MPI. Exitting.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &n_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// >>>
	
	printf("I am rank %d out of %d.\n", rank+1, n_tasks);

	open_files();
	pre_inference = clock();
	for(int i=rank; i<NUM_TESTS; i = i+n_tasks){
		get_char_float( i, (char_float*)&input, &label);
		assert( label >= 0 && label <= 10 );

		entry(input, result);
		int res = find_max(result);
		//printf("The input image corresponds to a: %d\n", res);
		//printf("Expected label:%d\n", label);
		if( label == res ){
			correct++;
			correct_bins[label]++;
		}
		else {
			wrong++;
			wrong_bins[label]++;
		}
	}
	pos_inference = clock();
	double inference_time = (double)(pos_inference-pre_inference)/CLOCKS_PER_SEC;
	printf("%.5lf\n", inference_time);
	
	rc = MPI_Reduce(&correct, &correct_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	rc = MPI_Reduce(&wrong, &wrong_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	for(int i=0; i<10; i++){
		rc = MPI_Reduce(correct_bins+i, correct_bins_sum+i, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		rc = MPI_Reduce(wrong_bins+i, wrong_bins_sum+i, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	if(rank==0){
		double max_inference_time = inference_time;
		correct = correct_sum;
		wrong = wrong_sum;
		for(int i=0; i<10; i++){
			correct_bins[i] = correct_bins_sum[i];
			wrong_bins[i] = wrong_bins_sum[i];
		}
		for(int i=0; i<n_tasks-1; i++){
			rc = MPI_Recv(&inference_time, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if(inference_time>max_inference_time)
				max_inference_time = inference_time;
		}
		printf("Model inference time was: %.6lf\n", max_inference_time/OMP_NUM_THREADS);
		printf("Overall accuracy: %d / %d  = %.1f accuracy\n", correct, correct+wrong, (float)(correct*100.0)/(correct+wrong));
		for(int i=0; i<10; i++) {
			printf("\tAccuracy for label %d: %f%%\n", i, (100.0 * correct_bins[i])/(correct_bins[i]+wrong_bins[i]));
		}
	} else {
		rc = MPI_Send(&inference_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}

int find_max(float result[1][10]){
	float max = 0.0;
	int index;

	//printf("Results:\n");
        for(int i =0; i<10; i++){
		if(result[0][i]>max){
		       	max = result[0][i];
			index = i;
		}
		//printf("%d->%f\t", i, result[0][i]);
	}
	//printf("\n");
	return index;
}

