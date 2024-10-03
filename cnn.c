#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <time.h>

typedef struct {
    int dims;
    int *shape;
    float *data;
} Tensor;

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride;
    int padding;
    float *weights;
    float *bias;
} ConvLayer;

typedef struct {
    int size;
    int stride;
} PoolLayer;

typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *bias;
} FCLayer;

typedef struct {
    ConvLayer *conv1;
    PoolLayer *pool1;
    ConvLayer *conv2;
    PoolLayer *pool2;
    FCLayer *fc1;
    FCLayer *fc2;
} CNN;

Tensor* create_tensor(int dims, int *shape) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    t->dims = dims;
    t->shape = (int*)malloc(dims * sizeof(int));
    memcpy(t->shape, shape, dims * sizeof(int));
    int size = 1;
    for(int i = 0; i < dims; i++) size *= shape[i];
    t->data = (float*)aligned_alloc(32, size * sizeof(float));
    memset(t->data, 0, size * sizeof(float));
    return t;
}

void free_tensor(Tensor *t) {
    free(t->shape);
    free(t->data);
    free(t);
}

void initialize_tensor(Tensor *t, float scale) {
    int size = 1;
    for(int i = 0; i < t->dims; i++) size *= t->shape[i];
    #pragma omp parallel for
    for(int i = 0; i < size; i++) t->data[i] = ((float)rand() / RAND_MAX) * scale;
}

ConvLayer* create_conv_layer(int in_channels, int out_channels, int kernel_h, int kernel_w, int stride, int padding) {
    ConvLayer *layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_h = kernel_h;
    layer->kernel_w = kernel_w;
    layer->stride = stride;
    layer->padding = padding;
    int weight_size = out_channels * in_channels * kernel_h * kernel_w;
    layer->weights = (float*)aligned_alloc(32, weight_size * sizeof(float));
    initialize_tensor(create_tensor(1, (int[]){weight_size}), 0.1f);
    for(int i = 0; i < weight_size; i++) layer->weights[i] = ((float)rand() / RAND_MAX) * sqrtf(2.0f / (in_channels * kernel_h * kernel_w));
    layer->bias = (float*)aligned_alloc(32, out_channels * sizeof(float));
    initialize_tensor(create_tensor(1, (int[]){out_channels}), 0.01f);
    for(int i = 0; i < out_channels; i++) layer->bias[i] = 0.0f;
    return layer;
}

FCLayer* create_fc_layer(int input_size, int output_size) {
    FCLayer *layer = (FCLayer*)malloc(sizeof(FCLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = (float*)aligned_alloc(32, input_size * output_size * sizeof(float));
    initialize_tensor(create_tensor(1, (int[]){input_size * output_size}), 0.1f);
    for(int i = 0; i < input_size * output_size; i++) layer->weights[i] = ((float)rand() / RAND_MAX) * sqrtf(2.0f / input_size);
    layer->bias = (float*)aligned_alloc(32, output_size * sizeof(float));
    initialize_tensor(create_tensor(1, (int[]){output_size}), 0.01f);
    for(int i = 0; i < output_size; i++) layer->bias[i] = 0.0f;
    return layer;
}

PoolLayer* create_pool_layer(int size, int stride) {
    PoolLayer *pool = (PoolLayer*)malloc(sizeof(PoolLayer));
    pool->size = size;
    pool->stride = stride;
    return pool;
}

CNN* create_cnn() {
    CNN *cnn = (CNN*)malloc(sizeof(CNN));
    cnn->conv1 = create_conv_layer(1, 8, 3, 3, 1, 1);
    cnn->pool1 = create_pool_layer(2, 2);
    cnn->conv2 = create_conv_layer(8, 16, 3, 3, 1, 1);
    cnn->pool2 = create_pool_layer(2, 2);
    cnn->fc1 = create_fc_layer(16 * 7 * 7, 128);
    cnn->fc2 = create_fc_layer(128, 10);
    return cnn;
}

void free_cnn(CNN *cnn) {
    free(cnn->conv1->weights);
    free(cnn->conv1->bias);
    free(cnn->conv1);
    free(cnn->pool1);
    free(cnn->conv2->weights);
    free(cnn->conv2->bias);
    free(cnn->conv2);
    free(cnn->pool2);
    free(cnn->fc1->weights);
    free(cnn->fc1->bias);
    free(cnn->fc1);
    free(cnn->fc2->weights);
    free(cnn->fc2->bias);
    free(cnn->fc2);
    free(cnn);
}

void relu(Tensor *input, Tensor *output) {
    int size = 1;
    for(int i = 0; i < input->dims; i++) size *= input->shape[i];
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        output->data[i] = fmaxf(0.0f, input->data[i]);
    }
}

void softmax(Tensor *input, Tensor *output) {
    int batch = input->shape[0];
    int classes = input->shape[1];
    #pragma omp parallel for
    for(int i = 0; i < batch; i++) {
        float max_val = input->data[i * classes];
        for(int j = 1; j < classes; j++) {
            if(input->data[i * classes + j] > max_val) max_val = input->data[i * classes + j];
        }
        float sum = 0.0f;
        for(int j = 0; j < classes; j++) {
            output->data[i * classes + j] = expf(input->data[i * classes + j] - max_val);
            sum += output->data[i * classes + j];
        }
        for(int j = 0; j < classes; j++) {
            output->data[i * classes + j] /= sum;
        }
    }
}

float cross_entropy_loss(Tensor *pred, Tensor *target) {
    float loss = 0.0f;
    int batch = pred->shape[0];
    int classes = pred->shape[1];
    #pragma omp parallel for reduction(+:loss)
    for(int i = 0; i < batch; i++) {
        for(int j = 0; j < classes; j++) {
            loss -= target->data[i * classes + j] * logf(fmaxf(pred->data[i * classes + j], 1e-7f));
        }
    }
    return loss / batch;
}

void matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    #pragma omp parallel for
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[j*K + k];
            }
            C[i*N + j] = sum;
        }
    }
}

void conv_forward(const ConvLayer *layer, const Tensor *input, Tensor *output) {
    int in_channels = layer->in_channels;
    int out_channels = layer->out_channels;
    int kernel_h = layer->kernel_h;
    int kernel_w = layer->kernel_w;
    int stride = layer->stride;
    int padding = layer->padding;
    int in_height = input->shape[2];
    int in_width = input->shape[3];
    int out_height = (in_height - kernel_h + 2 * padding) / stride + 1;
    int out_width = (in_width - kernel_w + 2 * padding) / stride + 1;
    #pragma omp parallel for collapse(3)
    for(int oc = 0; oc < out_channels; oc++) {
        for(int oh = 0; oh < out_height; oh++) {
            for(int ow = 0; ow < out_width; ow++) {
                float sum = layer->bias[oc];
                for(int ic = 0; ic < in_channels; ic++) {
                    for(int kh = 0; kh < kernel_h; kh++) {
                        for(int kw = 0; kw < kernel_w; kw++) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if(ih >=0 && ih < in_height && iw >=0 && iw < in_width) {
                                sum += input->data[ic*in_height*in_width + ih*in_width + iw] *
                                       layer->weights[oc*in_channels*kernel_h*kernel_w + ic*kernel_h*kernel_w + kh*kernel_w + kw];
                            }
                        }
                    }
                }
                output->data[oc*out_height*out_width + oh*out_width + ow] = sum;
            }
        }
    }
}

void pool_forward(const PoolLayer *pool, const Tensor *input, Tensor *output) {
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];
    int pool_size = pool->size;
    int stride = pool->stride;
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    #pragma omp parallel for collapse(3)
    for(int c = 0; c < in_channels; c++) {
        for(int h = 0; h < out_height; h++) {
            for(int w = 0; w < out_width; w++) {
                float max_val = -INFINITY;
                for(int ph = 0; ph < pool_size; ph++) {
                    for(int pw = 0; pw < pool_size; pw++) {
                        int ih = h * stride + ph;
                        int iw = w * stride + pw;
                        if(ih < in_height && iw < in_width) {
                            float val = input->data[c*in_height*in_width + ih*in_width + iw];
                            if(val > max_val) max_val = val;
                        }
                    }
                }
                output->data[c*out_height*out_width + h*out_width + w] = max_val;
            }
        }
    }
}

void fc_forward(const FCLayer *layer, const Tensor *input, Tensor *output) {
    matmul(input->data, layer->weights, output->data, input->shape[0], layer->input_size, layer->output_size);
    int size = input->shape[0] * layer->output_size;
    #pragma omp parallel for
    for(int i = 0; i < size; i++) output->data[i] += layer->bias[i % layer->output_size];
}

void forward_pass(CNN *cnn, Tensor *input, Tensor *output) {
    Tensor *conv1_out = create_tensor(4, (int[]){input->shape[0], cnn->conv1->out_channels, input->shape[2], input->shape[3]});
    conv_forward(cnn->conv1, input, conv1_out);
    Tensor *relu1 = create_tensor(4, conv1_out->shape);
    relu(conv1_out, relu1);
    free_tensor(conv1_out);
    Tensor *pool1_out = create_tensor(4, (int[]){relu1->shape[0], relu1->shape[1], relu1->shape[2]/cnn->pool1->stride, relu1->shape[3]/cnn->pool1->stride});
    pool_forward(cnn->pool1, relu1, pool1_out);
    free_tensor(relu1);
    Tensor *conv2_out = create_tensor(4, (int[]){pool1_out->shape[0], cnn->conv2->out_channels, pool1_out->shape[2], pool1_out->shape[3]});
    conv_forward(cnn->conv2, pool1_out, conv2_out);
    free_tensor(pool1_out);
    Tensor *relu2 = create_tensor(4, conv2_out->shape);
    relu(conv2_out, relu2);
    free_tensor(conv2_out);
    Tensor *pool2_out = create_tensor(4, (int[]){relu2->shape[0], relu2->shape[1], relu2->shape[2]/cnn->pool2->stride, relu2->shape[3]/cnn->pool2->stride});
    pool_forward(cnn->pool2, relu2, pool2_out);
    free_tensor(relu2);
    Tensor *flatten = create_tensor(2, (int[]){pool2_out->shape[0], pool2_out->shape[1]*pool2_out->shape[2]*pool2_out->shape[3]});
    memcpy(flatten->data, pool2_out->data, pool2_out->shape[1]*pool2_out->shape[2]*pool2_out->shape[3]*sizeof(float));
    free_tensor(pool2_out);
    Tensor *fc1_out = create_tensor(2, (int[]){flatten->shape[0], cnn->fc1->output_size});
    fc_forward(cnn->fc1, flatten, fc1_out);
    free_tensor(flatten);
    Tensor *relu3 = create_tensor(2, fc1_out->shape);
    relu(fc1_out, relu3);
    free_tensor(fc1_out);
    Tensor *fc2_out = create_tensor(2, (int[]){relu3->shape[0], cnn->fc2->output_size});
    fc_forward(cnn->fc2, relu3, fc2_out);
    free_tensor(relu3);
    softmax(fc2_out, output);
    free_tensor(fc2_out);
}

int main() {
    srand(time(NULL));
    CNN *cnn = create_cnn();
    int shape_input[4] = {1, 1, 28, 28};
    Tensor *input = create_tensor(4, shape_input);
    initialize_tensor(input, 0.1f);
    Tensor *output = create_tensor(2, (int[]){1, 10});
    forward_pass(cnn, input, output);
    Tensor *target = create_tensor(2, (int[]){1, 10});
    for(int i = 0; i < 10; i++) target->data[i] = 0.0f;
    target->data[3] = 1.0f; // Example target class
    float loss = cross_entropy_loss(output, target);
    printf("Loss: %f\n", loss);
    free_tensor(input);
    free_tensor(output);
    free_tensor(target);
    free_cnn(cnn);
    return 0;
}
