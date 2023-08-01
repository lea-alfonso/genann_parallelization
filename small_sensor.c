#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "genann.h"

int main(int argc, char *argv[])
{
    printf("GENANN example 1.\n");
    printf("Train a small ANN to the XOR function using backpropagation.\n");

    /* This will make the neural network initialize differently each run. */
    /* If you don't get a good result, try again for a different result. */
    srand(time(0));

    /* Input and expected output data for the XOR function. */
    double input[10][1] = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}};
    int rows = sizeof(input) / sizeof(input[0]);

    double output[10] = {1,2,3,4,5,6,7,8,9,10};
    int i;

    // Normalize the input data to [0, 1]
    double max_input = 10.0;
    for (i = 0; i < rows; ++i) {
        input[i][0] /= max_input;
    }

    // Normalize the output data to [0, 1]
    double max_output = 10.0;
    for (i = 0; i < rows; ++i) {
        output[i] /= max_output;
    }

    /* New network with 1 input,
     * 1 hidden layer of 5 neurons,
     * and 1 output. */
    genann *ann = genann_init(1, 1, 10, 1);

    /* Train on the data points many times. */
    for (i = 0; i < 100; ++i) {
        for (int j = 0; j < rows; ++j) {
            genann_train(ann, input[j], output + j, 0.15);
        }
    }

    double input_test[20][1] = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20}};
    int rows_test = sizeof(input_test) / sizeof(input_test[0]);

    double output_test[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    double max_input_test = 20.0;
    for (i = 0; i < rows_test; ++i) {
        input_test[i][0] /= max_input_test;
    }

    // Normalize the output data to [0, 1]
    double max_output_test = 20.0;
    for (i = 0; i < rows_test; ++i) {
        output_test[i] /= max_output_test;
    }

    /* New network with 1 input,
     * 1 hidden layer of 5 neurons,
     * and 1 output. */

    /* Run the network and see what it predicts. */
    for (int j = 0; j < rows_test; ++j) {
        // Denormalize the output back to the original scale [0, 10]
        double predicted_output = *genann_run(ann, input_test[j]) * max_output_test;
        printf("Output for [%1.f] is %f.\n", input_test[j][0] * max_input_test, predicted_output);
    }


    genann_free(ann);
    return 0;
}