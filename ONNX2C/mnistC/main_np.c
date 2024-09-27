#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "load.h"
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
        int correct=0;
        int wrong=0;
        int correct_bins[10]={0};
        int wrong_bins[10]={0};

        open_files();
        pre_inference = clock();
        for(int i=0; i<NUM_TESTS; i = i+1){
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

        printf("Model inference time was: %.6lf\n", inference_time/OMP_NUM_THREADS);
        printf("Overall accuracy: %d / %d  = %.1f accuracy\n", correct, correct+wrong, (float)(correct*100.0)/(correct+wrong));
        for(int i=0; i<10; i++) {
                printf("\tAccuracy for label %d: %f%%\n", i, (100.0 * correct_bins[i])/(correct_bins[i]+wrong_bins[i]));
	}
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

