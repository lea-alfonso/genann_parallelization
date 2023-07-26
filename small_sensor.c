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

    /* Input and expected out data for the XOR function. */
    const double input[4][1] = {{1},{2},{3},{4}};
    const double output[4] = {1,2,3,4};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    genann *ann = genann_init(1, 2, 3, 1);

    /* Train on the four labeled data points many times. */
    for (i = 0; i < 1000; ++i) {
        genann_train(ann, input[0], output + 0, 0.3);
        genann_train(ann, input[1], output + 1, 0.3);
        genann_train(ann, input[1], output + 2, 0.3);
        genann_train(ann, input[1], output + 3, 0.3);
    }

    /* Run the network and see what it predicts. */
    const double hardcoded_outputs[3][1] = {{3},{4},{5}};
    printf("Output for [%1.f] is %1.f.\n", input[0][0], *genann_run(ann, input[0]));
    printf("Output for [%1.f] is %1.f.\n", input[1][0], *genann_run(ann, input[1]));
    printf("Output for [%1.d] is %1.f.\n",3, *genann_run(ann, hardcoded_outputs[0]));
    printf("Output for [%1.d] is %1.f.\n", 4, *genann_run(ann, hardcoded_outputs[1]));
    printf("Output for [%1.d] is %1.f.\n", 5, *genann_run(ann, hardcoded_outputs[2]));

    genann_free(ann);
    return 0;
}