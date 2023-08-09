#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "genann.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */

// const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

int load_data(double **ref_general_input,int nodes) {
    FILE *in = fopen("example/training_data_0.csv", "r");
    if (!in) {
        printf("Could not open file\n");
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    unsigned long samples = 0;
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);
    printf("Loading %ld data points \n", samples);

    /* Allocate memory for input*/
    *ref_general_input = malloc(sizeof(double) * samples * (5 + nodes));

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
        double *p = *ref_general_input + i * (5 + nodes);
        
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
        double *p = *ref_general_input + i * (5 + nodes);
        for (j = 0; j < (5+nodes); ++j) {
            if (max_values[j] == min_values[j]) {
                p[j] = 0;  
            } else {
                p[j] = (p[j] - min_values[j]) / (max_values[j] - min_values[j]);
            }
        }
    }
    return samples;
}

void reorganize_data(double **input,int samples, int nodes,int node_index,double **node_input,double **node_output) {

    for (int i = 0; i < samples; ++i) {
        double *gral_index = *input + (i * (5 + nodes));
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

void shuffle_data(double *input, double *class, int samples) {
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

int main(int argc, char *argv[]) {
    int nodes = 7;
    int number_of_cores = 1;
    int loops = 500;
    double train_test_percentage = 0.80;
    double accepted_relative_error = 0.20;
    double* general_input;
    int samples = load_data(&general_input,nodes);

    int train_test_index = samples * train_test_percentage;
    double results[nodes];
    int i,j,k,l;

    for(i=0; i < nodes;++i) {

        /* We create mem space for the input and output of each node*/
        double *node_input = malloc(sizeof(double) * samples * (5 + nodes-1));
        double *node_output = malloc(sizeof(double) * samples );

        //First we need to remove the feature we're trying to predict from the input and move them to the output
        reorganize_data(&general_input,samples,nodes,i,&node_input,&node_output);

        // for (j=0; j < 1 ; ++j) {
        //     double *node_index = node_input +(j * (5+nodes -1));
        //     for (k = 0; k < (5+nodes-1); ++k) {
        //         printf("%f, ",node_index[k]);
        //     }
        //     printf("%f\n",node_output[j]);

        // }

        //We create the NN 
        genann *ann = genann_init(5 + nodes - 1, 1, 45, 1);

        //Train
        for (j = 0; j < loops; ++j) {
            for (k = 0; k < train_test_index; ++k) {
                genann_train(ann, node_input + k*(5 + (nodes - 1)), node_output + k, .01);
            }
            printf("Loop %d \n",j);
        }

        //Test
        int correct = 0;
        for (l = train_test_index; l < samples; ++l) {
            if (fabs(node_output[l] - *genann_run(ann,node_input + l*(5 + (nodes - 1))) ) < (node_output[l]* accepted_relative_error) ) {
                ++correct;
            }
        }
        results[i] = (double)correct / (samples - train_test_index) * 100.0;
        printf("%d/%d correct (%0.1f%%).\n", correct, samples - train_test_index, (double)correct / (samples - train_test_index) * 100.0);

        char filename[100];  // You need to allocate enough space for the filename

        // Format the integer value into a string and concatenate it with the filename
        sprintf(filename, "output/node_%d.ann", i);
        FILE *file = fopen(filename, "w");

        if (file == NULL) {
            printf("Error opening the file.\n");
            return 1;
        }
        genann_write(ann,file);
        fclose(file);

        free(node_input);
        free(node_output);
        genann_free(ann);
    }
    free(general_input);
    return 0;
}
