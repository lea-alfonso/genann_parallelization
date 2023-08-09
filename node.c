#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "genann.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */

// const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
int nodes = 7;
int number_of_cores = 1;
int loops = 500;
double train_test_percentage = 0.80;
double accepted_relative_error = 0.20;
double* general_input;
unsigned long samples;
int train_test_index ;
double* results;

void load_data() {
    FILE *in = fopen("example/training_data_0.csv", "r");
    if (!in) {
        printf("Could not open file\n");
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    samples = 0;
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);
    printf("Loading %ld data points \n", samples);

    /* Allocate memory for input*/
    general_input = malloc(sizeof(double) * samples * (5 + nodes));

    /* Allocate memory for min and max values of each feature for normalization*/
    double min_values[5+nodes];
    double max_values[5+nodes];
    for (int i = 0; i < (5+nodes); ++i) {
        min_values[i] = 10000; // Initialize to positive max
        max_values[i] = 0; // Initialize to negative min
    }
    
    /* Read the file into our array. */
    unsigned int i;
    int j;
    for (i = 0; i < samples; ++i) {
        double *p = general_input + i * (5 + nodes);
        
        if (fgets(line, 1024, in) == NULL) {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");

        for (j = 0; j < (5+nodes); ++j) {
            p[j] = atof(split);

            // We update the min,max values
            if (p[j] < min_values[j]) {
                min_values[j] = p[j];
            }
            if (p[j] > max_values[j]) {
                max_values[j] = p[j];
            }
            split = strtok(0, ",");
        }

    }
    fclose(in);
    
     // Normalize the input data using min-max normalization
    for (i = 0; i < samples; ++i) {
        double *p = general_input + i * (5 + nodes);
        for (j = 0; j < (5+nodes); ++j) {
            if (max_values[j] == min_values[j]) {
                p[j] = 0;  
            } else {
                p[j] = (p[j] - min_values[j]) / (max_values[j] - min_values[j]);
            }
        }
    }
}


void reorganize_data(int node_index,double **node_input,double **node_output) {

    for (int i = 0; i < samples; ++i) {
        double *gral_index = general_input + (i * (5 + nodes));
        double *input_index = *node_input + (i * (5 + nodes-1));
        double *output_index = *node_output + i;
        
        for (int j = 0; j < (5 + nodes);++j) {
            if (j < 5 + node_index) {
                input_index[j] = gral_index[j];
            } else if (j == 5 + node_index) {
                output_index[0] = gral_index[j];
            } else if (j > 5 + node_index){
                input_index[j-1] = gral_index[j];
            }
        }
    }
}

void shuffle_data(double *input, double *class) {
    double temp;
    for (int i = samples - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        if (i != j) {
            // Swap input samples
            for (int k = 0; k < 7; ++k) {
                temp = input[i * 7 + k];
                input[i * 7 + k] = input[j * 7 + k];
                input[j * 7 + k] = temp;
            }
            // Swap output samples
            temp = class[i];
            class[i] = class[j];
            class[j] = temp;
        }
    }
}
void *thread_function(void *arg) {
    int i = *(int *)arg;
    printf("Starting thread %d\n",i);
    int j,k,l;
    double *node_input = malloc(sizeof(double) * samples * (5 + nodes - 1));
    double *node_output = malloc(sizeof(double) * samples);
    reorganize_data(i,&node_input,&node_output);

    
    //We create the NN 
    genann *ann = genann_init(5 + nodes - 1, 1, 45, 1);

    //Train
    for (j = 0; j < loops; ++j) {
        for (k = 0; k < train_test_index; ++k) {
            genann_train(ann, node_input + k*(5 + (nodes - 1)), node_output + k, .01);
        }
    }

    //Test
    int correct = 0;
    for (l = train_test_index; l < samples; ++l) {
        if (fabs(node_output[l] - *genann_run(ann,node_input + l*(5 + (nodes - 1))) ) < (node_output[l]* accepted_relative_error) ) {
            ++correct;
        }
    }
    results[i] = (double)correct / (samples - train_test_index) * 100.0;

    char filename[100];  // You need to allocate enough space for the filename

    // Format the integer value into a string and concatenate it with the filename
    sprintf(filename, "output/node_%d.ann", i);
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        printf("Error opening the file.\n");
        pthread_exit(NULL);
    }
    genann_write(ann,file);
    fclose(file);

    free(node_input);
    free(node_output);
    genann_free(ann);

    printf("Thread %d finished!\n",i);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    int i;
    load_data();
    train_test_index = samples * train_test_percentage;
    results = malloc(sizeof(double) * nodes);
    pthread_t threads[nodes]; // Array to store thread IDs
    int thread_args[nodes]; // Array to store thread-specific arguments

    for(i=0; i < nodes;++i) {
        thread_args[i] = i;
        // Create thread
        if (pthread_create(&threads[i], NULL, thread_function, &thread_args[i]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            return 1;
        }
    }
    // Wait for all threads to finish
    for (i = 0; i < nodes; ++i) {
        pthread_join(threads[i], NULL);
    }
    printf("All threads finished!\nResults are:\n");
    for (i = 0; i< nodes; ++i) {
        printf("\t Thread %d: (%0.1f%%) accuracy\n",i,results[i]);
    }
    return 0;
}
