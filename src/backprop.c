#include "../include/backprop.h"

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

/*** Return random number between 0.0 and 1.0 ***/
float drnd()
{
  return ((float) rand() / (float) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
float dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(x)
float x;
{
  float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(n)
int n;
{
  float *new;

  new = (float *) malloc ((unsigned) (n * sizeof (float)));
  if (new == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return (new);
}


/*** Allocate 2d array of floats ***/

float **alloc_2d_dbl(m, n)
int m, n;
{
  int i;
  float **new;

  new = (float **) malloc ((unsigned) (m * sizeof (float *)));
  if (new == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    new[i] = alloc_1d_dbl(n);
  }

  return (new);
}


void bpnn_randomize_weights(w, m, n)
float **w;
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[i][j] = (float) rand()/RAND_MAX;
    //  w[i][j] = dpn1();
    }
  }
}

void bpnn_randomize_row(w, m)
float *w;
int m;
{
	int i;
	for (i = 0; i <= m; i++) {
     //w[i] = (float) rand()/RAND_MAX;
	 w[i] = 0.1;
    }
}

void bpnn_zero_weights(w, m, n)
float **w;
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}

void bpnn_initialize(int seed)
{
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}

BPNN *bpnn_internal_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}

void bpnn_free(net)
BPNN *net;
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i <= n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}

int load(BPNN *net, int layer_size, int *counter)
{
    float *units = net->input_units;

    int max_iterations_per_cycle = 1000000;

    int iterations = layer_size - (*counter - 1);
    if(max_iterations_per_cycle < iterations) iterations = max_iterations_per_cycle;

    int j;

    printf("Computing with 1 thread\n");

    for(int n = 0; n < iterations; n++) {

        j = *counter + n;
        units[j] = (float) rand()/RAND_MAX;
    }

    *counter += iterations;

    printf("Counter = %d\nLayer_size = %d\nIterations = %d\n", *counter, layer_size, iterations);

    if(*counter > layer_size) {

        *counter = 1;
        return 1;
    }

    return 0;
}

int bpnn_layerforward(l1, l2, conn, n1, n2, thread_num, counter)
float *l1, *l2, **conn;
int n1, n2, thread_num, *counter;
{
    int max_iterations_per_cycle = thread_num;

    int iterations = n2 - (*counter - 1);
    if(max_iterations_per_cycle < iterations) iterations = max_iterations_per_cycle;

    int iterations_per_thread = iterations / thread_num;
    while(iterations_per_thread == 0 && thread_num > 1) {

        thread_num--;
        iterations_per_thread = iterations / thread_num;
    }

    float sum;
    int j;

    /*** Set up thresholding unit ***/
    l1[0] = 1.0;

    //parallel work
    omp_set_num_threads(thread_num);
    #pragma omp parallel private(sum, j)
    {
    #pragma omp single
        {
            thread_num = omp_get_num_threads();
            printf("Computing with %d threads\n", thread_num);
        }

        int tid = omp_get_thread_num();

        for(int n = 0; n < iterations_per_thread; n++) {

            j = *counter + n + iterations_per_thread * tid;
            if(j > n2) break;

            sum = 0.0;
            for (int k = 0; k <= n1; k++) {

                sum += conn[k][j] * l1[k];
            }
            l2[j] = squash(sum);
        }
    }

    *counter += iterations_per_thread * thread_num;

    printf("Counter = %d\nN2 = %d\nIterations = %d\nIterations_per_thread = %d\n", *counter, n2, iterations, iterations_per_thread);

    if(*counter > n2) {

        *counter = 1;
        return 1;
    }

    return 0;
}

void bpnn_output_error(delta, target, output, nj, err)  
float *delta, *target, *output, *err;
int nj;
{
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

void bpnn_hidden_error(delta_h,   
					   nh, 
					   delta_o, 
					   no, 
					   who, 
					   hidden, 
					   err)
float *delta_h, *delta_o, *hidden, **who, *err;
int nh, no;
{
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}


int bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw, thread_num, counter)
float *delta, *ly, **w, **oldw;
int ndelta, nly, thread_num, *counter;
{
    int max_iterations_per_cycle = thread_num;

    int iterations = ndelta - (*counter - 1);
    if (max_iterations_per_cycle < iterations) iterations = max_iterations_per_cycle;

    int iterations_per_thread = iterations / thread_num;
    while(iterations_per_thread == 0 && thread_num > 1) {

        thread_num--;
        iterations_per_thread = iterations / thread_num;
    }

    float new_dw;
    int j;
    ly[0] = 1.0;

    omp_set_num_threads(thread_num);
    #pragma omp parallel private(new_dw, j)
    {
    #pragma omp single
        {
            thread_num = omp_get_num_threads();
            printf("Computing with %d threads\n", thread_num);
        }

        int tid = omp_get_thread_num();

        for (int n = 0; n < iterations_per_thread; n++) {

            j = *counter + n + iterations_per_thread * tid;
            if (j > ndelta) break;

            for (int k = 0; k <= nly; k++) {
                new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
                w[k][j] += new_dw;
                oldw[k][j] = new_dw;
            }
        }
    }

    *counter += iterations_per_thread * thread_num;

    printf("Counter = %d\nNdelta = %d\nIterations = %d\nIterations_per_thread = %d\n", *counter, ndelta, iterations,
           iterations_per_thread);

    if (*counter > ndelta) {

        *counter = 1;
        return 1;
    }

    return 0;
}
