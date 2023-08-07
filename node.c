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

int load_data(const char* sensor_data,double *general_input,const int nodes) {
    /* Load the iris data-set. */
    FILE *in = fopen(sensor_data, "r");
    if (!in) {
        printf("Could not open file: %s\n", sensor_data);
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    int samples = 0;
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d data points from %s\n", samples, sensor_data);

    /* Allocate memory for input*/
    general_input = malloc(sizeof(double) * samples * (5 + nodes));

    /* Read the file into our arrays. */
    int i, j;
    for (i = 0; i < samples; ++i) {
        // We have 6 integer features and only one integer output
        double *p = general_input + i * (5 + nodes);
        
        if (fgets(line, 1024, in) == NULL) {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        for (j = 0; j < (5+nodes); ++j) {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

    }
    fclose(in);
    return samples;
}

void reorganize_data(double* input,int samples, int nodes,int node_index,double* node_input,double* node_output) {
    /* We create mem space for the input and output of each node*/
    node_input = malloc(sizeof(double) * samples * (5 + (nodes-1)));
    node_output = malloc(sizeof(double) * samples );

    for (int i = 0; i < samples; ++i) {
        double *gral_index = input + i * (5 + (nodes));
        double *input_index = node_input + i * (5 + (nodes-1));
        double *output_index = node_output + i;
        
        for (int j = 0; j < 5 + (nodes-1);++j) {
            if (j != node_index) {
                input_index[j] = gral_index[j];
            } else {
                output_index[0] = gral_index[j];
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
void master_node() {
    const char *sensor_data = "example/training_data_0.csv";
    int nodes = 10;
    int loops = 500;
    double train_test_percentage = 0.80;
    double accepted_relative_error = 0.20;
    double* general_input;
    int samples = load_data(sensor_data,general_input,nodes);
    int train_test_index = samples * train_test_percentage;
    int i,j,k;

    for(i=0; i < nodes;++i) {

        //First we need to remove the feature we're trying to predict from the input
        double *node_input;
        double *node_output;
        reorganize_data(general_input,samples,nodes,i,node_input,node_output);

        //We create the NN 
        genann *ann = genann_init(6, 2, 15, 1);

        //Train
        for (j = 0; j < loops; ++i) {
            for (k = 0; k < train_test_index; ++k) {
                genann_train(ann, node_input + k*(5 + (nodes - 1)), node_output + k, .01);
            }
        }

        //Test
        int correct = 0;
        for (k = train_test_index; k < samples; ++j) {
            if (fabs(node_output[k] - *genann_run(ann,node_input + k*(5 + (nodes - 1))) ) < (node_output[k]* accepted_relative_error) ) {
                ++correct;
            }
        }

        printf("%d/%d correct (%0.1f%%).\n", correct, samples - train_test_index, (double)correct / (samples - train_test_index) * 100.0);
    }
}
int main(int argc, char *argv[]) {
    master_node();
    return 0;
}
