#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "genann.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */
const char *sensor_data = "example/stripped_data.csv";

double *input, *class;
int samples;
// const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

void load_data(double *min_values, double *max_values) {
    /* Load the iris data-set. */
    FILE *in = fopen("example/stripped_data.csv", "r");
    if (!in) {
        printf("Could not open file: %s\n", sensor_data);
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d data points from %s\n", samples, sensor_data);

    /* Allocate memory for input and output data. */
    input = malloc(sizeof(double) * samples * 6);
    class = malloc(sizeof(double) * samples );
    /* Initialize min and max arrays to keep track of min and max values for each feature. */
    for (int i = 0; i < 7; ++i) {
        min_values[i] = 10000; // Initialize to positive infinity
        max_values[i] = 0; // Initialize to negative infinity
    }

    /* Read the file into our arrays. */
    int i, j;
    for (i = 0; i < samples; ++i) {
        // We have 6 integer features and only one integer output
        double *p = input + i * 6;
        double *c = class + i;
        
        if (fgets(line, 1024, in) == NULL) {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        for (j = 0; j < 6; ++j) {
            p[j] = atof(split);
            // Update min and max values for each feature
            if (p[j] < min_values[j]) {
                min_values[j] = p[j];
            }
            if (p[j] > max_values[j]) {
                max_values[j] = p[j];
            }
            split = strtok(0, ",");
        }
        c[0] = atof(split);
        if (c[0] < min_values[6]) {
            min_values[6] = c[0];
        }
        if (c[0] > max_values[6]) {
            max_values[6] = c[0];
        }

        // printf("Data point %d is %1.f %1.f %1.f %1.f %1.f %1.f ->  %f\n", i, p[0], p[1], p[2], p[3], p[4], p[5], c[0]); 
    }
    fclose(in);
     // Normalize the input data using min-max normalization
    for (i = 0; i < samples; ++i) {
        double *p = input + i * 6;
        double *c = class + i;
        for (j = 0; j < 6; ++j) {
            // printf("Min %f Max: %f \n",min_values[j],max_values[j]);
            if (max_values[j] == min_values[j]) {
                p[j] = (p[j] - min_values[j]);  
            } else {
                p[j] = (p[j] - min_values[j]) / (max_values[j] - min_values[j]);
            }
        }
        c[0] = (c[0] - min_values[6]) / (max_values[6] - min_values[6]);
        // printf("Normalized data point %d is %f %f %f %f %f %f ->  %f\n", i, p[0], p[1], p[2], p[3], p[4], p[5], c[0]); 
    }

}
void shuffle_data(double *input, double *class, int samples) {
    double temp;
    for (int i = samples - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        if (i != j) {
            // Swap input samples
            for (int k = 0; k < 6; ++k) {
                temp = input[i * 6 + k];
                input[i * 6 + k] = input[j * 6 + k];
                input[j * 6 + k] = temp;
            }
            // Swap output samples
            temp = class[i];
            class[i] = class[j];
            class[j] = temp;
        }
    }
}
int main(int argc, char *argv[])
{
    printf("GENANN example 4.\n");
    printf("Train an ANN on the stripped dataset using backpropagation.\n");

    srand(time(0));

    /* Load the data from file. */
    double min_values[7];
    double max_values[7];
    load_data(min_values,max_values);

    shuffle_data(input,class,samples);
    for (int i = 0; i < samples; ++i) {
        double *p = input + i * 6;
        double *c = class + i;
        // printf("Shuffled data point %d is %f %f %f %f %f %f ->  %f\n", i, p[0], p[1], p[2], p[3], p[4], p[5], c[0]);
    }
    /* 6 inputs.
     * 1 hidden layer(s) of 15 neurons.
     * 1 output 
     */
    genann *ann = genann_init(6, 1, 15, 1);

    int i, j;
    int loops = 1000;
    double train_test_percentage = 0.80;
    double accepted_relative_error = 0.20;
    int train_test_index = samples * train_test_percentage;
    /* Train the network with backpropagation. */
    printf("Training for %d loops over data samples %d.\n", loops,samples);
    for (i = 0; i < loops; ++i) {
        for (j = 0; j < train_test_index; ++j) {
            genann_train(ann, input + j*6, class + j, .005);
        }
        /* printf("%1.2f ", xor_score(ann)); */
    }

    int correct = 0;
    for (j = train_test_index; j < samples; ++j) {
        if (fabs(class[j] - *genann_run(ann, input + j*6)) < (class[j]* accepted_relative_error) ) {
            ++correct;
        }
        // printf("Desired/Output %f/%f,AbsDiff/Diff %f  -  %f \n",class[j],*genann_run(ann, input + j*6),fabs(class[j] - *genann_run(ann, input + j*6)),class[j] - *genann_run(ann, input + j*6));
    }

    printf("%d/%d correct (%0.1f%%).\n", correct, samples - train_test_index, (double)correct / (samples - train_test_index) * 100.0);



    genann_free(ann);
    free(input);
    free(class);

    return 0;
}
