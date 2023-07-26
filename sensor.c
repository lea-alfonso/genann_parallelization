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

void load_data() {
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
            split = strtok(0, ",");
        }
        split[strlen(split)-1] = 0;
        c[0] = atof(split);

        // printf("Data point %d is %d %d %d %d %d %d ->  %d\n", i, p[0], p[1], p[2], p[3], p[4], p[5], c[0]); 
    }

    fclose(in);
}


int main(int argc, char *argv[])
{
    printf("GENANN example 4.\n");
    printf("Train an ANN on the IRIS dataset using backpropagation.\n");

    srand(time(0));

    /* Load the data from file. */
    load_data();

    /* 6 inputs.
     * 1 hidden layer(s) of 15 neurons.
     * 1 output 
     */
    genann *ann = genann_init(6, 1, 15, 1);

    int i, j;
    int loops = 5000;

    /* Train the network with backpropagation. */
    printf("Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; ++i) {
        for (j = 0; j < samples; ++j) {
            genann_train(ann, input + j*6, class + j, .01);
        }
        /* printf("%1.2f ", xor_score(ann)); */
    }

    int correct = 0;
    for (j = 0; j < samples; ++j) {
        if (class[j] == *genann_run(ann, input + j*6)) {
            ++correct;
        }
    }

    printf("%d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);



    genann_free(ann);
    free(input);
    free(class);

    return 0;
}
