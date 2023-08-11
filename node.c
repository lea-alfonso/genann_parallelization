#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <pthread.h>
#include "genann.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */

// const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
int nodes = 7;
int num_cores;
int num_threads;
int problems_per_thread;
int loops = 500;
double train_test_percentage = 0.80;
double accepted_relative_error = 0.20;
double* general_input;
unsigned long samples;
int train_test_index ;
double* results;
double* train_times;
double* thread_times;
int* problem_trained_in_thread;

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
    struct timeval startThread, endThread;
    int node_index = (i * problems_per_thread);
    if (i <= nodes % num_threads) {
        node_index += i;
    } else {
        node_index += nodes % num_threads;
    }
    int problems_to_solve = problems_per_thread + (i < nodes % num_threads);
    printf("Starting Thread %d, will be trining problems %d to %d\n",i,node_index, node_index + problems_to_solve - 1);
    gettimeofday(&startThread, NULL);

    for (int problem_number = node_index; problem_number < node_index + problems_to_solve; ++problem_number) {
        int j,k,l;
        problem_trained_in_thread[problem_number] = i;
        double *node_input = malloc(sizeof(double) * samples * (5 + nodes - 1));
        double *node_output = malloc(sizeof(double) * samples);
        reorganize_data(problem_number,&node_input,&node_output);
        struct timeval start, end;

        //We create the NN
        genann *ann = genann_init(5 + nodes - 1, 1, 45, 1);

        gettimeofday(&start, NULL);
        //Train
        for (j = 0; j < loops; ++j) {
            for (k = 0; k < train_test_index; ++k) {
                genann_train(ann, node_input + k*(5 + (nodes - 1)), node_output + k, .01);
            }
        }
        gettimeofday(&end, NULL);
        train_times[problem_number] = end.tv_sec + end.tv_usec / 1e6 -
                            start.tv_sec - start.tv_usec / 1e6;

        //Test
        int correct = 0;
        for (l = train_test_index; l < samples; ++l) {
            if (fabs(node_output[l] - *genann_run(ann,node_input + l*(5 + (nodes - 1))) ) < (node_output[l]* accepted_relative_error) ) {
                ++correct;
            }
        }
        results[problem_number] = (double)correct / (samples - train_test_index) * 100.0;

        char filename[100];  // You need to allocate enough space for the filename

        // Format the integer value into a string and concatenate it with the filename
        sprintf(filename, "output/node_%d.ann", problem_number);
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
    }

    gettimeofday(&endThread, NULL);
    thread_times[i] = endThread.tv_sec + endThread.tv_usec / 1e6 - startThread.tv_sec - startThread.tv_usec / 1e6;

    printf("Thread %d finished!\n",i);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    int i;
    load_data();
    train_test_index = samples * train_test_percentage;
    num_cores = get_nprocs();
    num_threads = (nodes < num_cores) ? nodes : num_cores;
    problems_per_thread = (nodes < num_cores) ? 1 : (int)(nodes / num_threads);
    results = malloc(sizeof(double) * nodes);
    problem_trained_in_thread = malloc(sizeof(int) * nodes);
    train_times = malloc(sizeof(double) * nodes);
    thread_times = malloc(sizeof(double) * num_threads);
    struct timeval start, end;
    pthread_t threads[num_threads]; // Array to store thread IDs
    int thread_args[num_threads]; // Array to store thread-specific arguments
    printf("Number of available cores: %d\n",num_cores);
    printf("Creating %d threads with %d to %d problems each\n",num_threads, problems_per_thread, problems_per_thread + 1);
    for(i=0; i < num_threads;++i) {
        thread_args[i] = i;
        // Create thread
        if (pthread_create(&threads[i], NULL, thread_function, &thread_args[i]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            return 1;
        }
    }
    gettimeofday(&start, NULL);
    // Wait for all threads to finish
    for (i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    gettimeofday(&end, NULL);
    double time_spent = end.tv_sec + end.tv_usec / 1e6 -
                        start.tv_sec - start.tv_usec / 1e6;
    printf("All threads finished in a total time of %0.1f seconds!\nResults are:\n", time_spent);
    for (i = 0; i< nodes; ++i) {
        printf("\t Problem %d, Thread %d: (%0.1f%%) accuracy trained in %0.1f seconds\n",i,problem_trained_in_thread[i],results[i],train_times[i]);
    }
    for (i = 0; i< num_threads; ++i) {
        printf("\t Thread %d trained in %0.1f seconds\n",i,thread_times[i]);
    }
    return 0;
}
