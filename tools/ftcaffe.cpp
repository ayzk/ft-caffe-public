/*

Fault-tolerance features added by Kai Zhao. 2020 University of California, Riverside

All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <set>
//#include <cstdlib>
#include <cmath>
#include <cstdio>
#include "boost/algorithm/string.hpp"
#include "boost/make_shared.hpp"
#include "caffe/caffe.hpp"
#include "caffe/training_utils.hpp"
#include "caffe/util/performance.hpp"
#include "caffe/util/signal_handler.h"
//#include "caffe/util/abft_math_functions.hpp"
#include <emmintrin.h>
#include <pmmintrin.h>
#include <numeric>
#include "caffe/util/bbox_util.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MLSL
#include "caffe/multinode/async_param_server.hpp"
#include "caffe/multinode/mlsl.hpp"
#include "caffe/multinode/multi_sync.hpp"
#endif /* USE_MLSL */

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::Solver;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(model, "", "The model definition protocol buffer text file.");
DEFINE_int32(level, 0, "Optional; network level.");
DEFINE_string(stage, "",
              "Optional; network stages (not to be confused with phase), "
              "separated by ','.");
DEFINE_string(weights, "",
              "Optional; the pretrained weights to initialize finetuning, "
              "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(
        engine, "",
        "Optional; Engine sequence in format: engine:subengine_1,subengine_2,...");
DEFINE_string(val_dir, "", "Optional; Directory with validation pictures");
DEFINE_int32(val_begin, 0, "Optional; number of validation");
DEFINE_int32(val_end, 50000, "Optional; number of validation");
DEFINE_int32(val_stride, 1, "Optional; number of validation");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();

typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func)                           \
    namespace {                                              \
    class __Registerer_##func {                              \
       public: /* NOLINT */                                  \
        __Registerer_##func() { g_brew_map[#func] = &func; } \
    };                                                       \
    __Registerer_##func g_registerer_##func;                 \
    }

static BrewFunction GetBrewFunction(const caffe::string &name) {
    if (g_brew_map.count(name)) {
        return g_brew_map[name];
    } else {
        LOG(ERROR) << "Available caffe actions:";
        for (BrewMap::iterator it = g_brew_map.begin(); it != g_brew_map.end();
             ++it) {
            LOG(ERROR) << "\t" << it->first;
        }
        LOG(FATAL) << "Unknown action: " << name;
        return NULL;  // not reachable, just to suppress old compiler warnings.
    }
}

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float> *solver, const std::string &model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(","));
    for (int i = 0; i < model_names.size(); ++i) {
        LOG(INFO) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < solver->test_nets().size(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

float SSE(const long N, const float *p) {
    __m128 mmSum = _mm_setzero_ps();
    size_t i = 0;
    for (; i < ROUND_DOWN(N, 16); i += 16) {
        __m128 v0 = _mm_loadu_ps(p + i + 0);
        __m128 v1 = _mm_loadu_ps(p + i + 4);
        __m128 s01 = _mm_add_ps(v0, v1);

        __m128 v2 = _mm_loadu_ps(p + i + 8);
        __m128 v3 = _mm_loadu_ps(p + i + 12);
        __m128 s23 = _mm_add_ps(v2, v3);

        mmSum = _mm_add_ps(mmSum, s01);
        mmSum = _mm_add_ps(mmSum, s23);
    }
    // unrolled loop that adds up 4 elements at a time
    for (; i < ROUND_DOWN(N, 4); i += 4) {
        mmSum = _mm_add_ps(mmSum, _mm_loadu_ps(p + i));
    }

    // add up single values until all elements are covered
    for (; i < N; i++) {
        mmSum = _mm_add_ss(mmSum, _mm_load_ss(p + i));
    }

    // add up the four float values from mmSum into a single value and return
    mmSum = _mm_hadd_ps(mmSum, mmSum);
    mmSum = _mm_hadd_ps(mmSum, mmSum);
    return _mm_cvtss_f32(mmSum);
}

template<typename Dtype>
Dtype BlobSum(Blob<Dtype> *blob) {
    const Dtype *A = blob->cpu_data();
    long count = blob->count();
    Dtype sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel reduction(+ : sum)
    {
        long id = omp_get_thread_num();
        long nthrds = omp_get_num_threads();

        long block_low = id * count / nthrds;
        long block_high = (id + 1) * count / nthrds - 1;
        if (std::is_same<Dtype, float>::value) {
            sum += SSE(block_high - block_low + 1, A + block_low);
        } else {
            for (long i = block_low; i < block_high - block_low + 1; i++) {
                sum += A[i];
            }
        }
    }
#else
    if (std::is_same<Dtype, float>::value) {
        sum += SSE(count, A);
    } else {
        for (long i = 0; i < count; i++) {
            sum += A[i];
        }
    }
#endif
    return sum;
}

int caffe_getenv(const char *name) {
    int result = 0;
    char *buffer = getenv(name);
    if (buffer != NULL) {
        result = atoi(buffer);
    }

    return result;
}

// void caffe_cpu_gemv_full(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE
// TransA, const int M,
//                                const int N, const float alpha, const float
//                                *A, const int lda, const float *x, const int
//                                incX, const float beta, float *y, const int
//                                incY) {
//    cblas_sgemv(Layout, TransA, M, N, alpha, A, lda, x, incX, beta, y, incY);
//}

void compute_checksum_column_block(const int m, const int n, const float *A,
                                   const float beta, float *ck, int r0) {
    Timer timer;
    timer.Start();
    std::vector<float> R(m);
    if (r0 == 0) {
        std::fill(R.begin(), R.end(), 1);
    } else {
        std::iota(R.begin(), R.end(), r0);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, n, m, 1, R.data(),
                m, A, n, beta, ck, n);
}

void compute_checksum_row_block(const int m, const int n, const int k,
                                const float *A, const float beta, float *ck,
                                int r0) {
#ifdef _OPENMP
#pragma omp parallel for if (m > 1)
#endif

    for (int i = 0; i < m; i++) {
        compute_checksum_column_block(n, k, A + i * n * k, beta, ck + i * k,
                                      r0);
    }
}

template<typename Dtype>
class Operation {
public:
    Operation(caffe::LayerParameter layerParameter,
              shared_ptr<Blob<Dtype>> _bottom, vector<int> top_shape)
            : top(new Blob<Dtype>(top_shape)),
              checksum(new Blob<Dtype>(top_shape)) {
        type = 1;
        bottom = _bottom;
        layer = caffe::LayerRegistry<Dtype>::CreateLayer(layerParameter);
        layer->SetUp({_bottom.get()}, {top.get()});
    }

    Operation(caffe::LayerParameter layerParameter, Blob<Dtype> *_bottom,
              vector<int> top_shape)
            : top(new Blob<Dtype>(top_shape)),
              checksum(new Blob<Dtype>(top_shape)) {
        type = 2;
        bottom1 = _bottom;
        layer = caffe::LayerRegistry<Dtype>::CreateLayer(layerParameter);
        layer->SetUp({bottom1}, {top.get()});
    }

    Operation(vector<int> bottom_shape, vector<int> top_shape)
            : top(new Blob<Dtype>(top_shape)),
              bottom(new Blob<Dtype>(bottom_shape)),
              checksum(new Blob<Dtype>(top_shape)) {
        type = 3;
    }

    inline Dtype Forward() {
        if (type == 1) {
            return layer->Forward({bottom.get()}, {top.get()});
        } else if (type == 2) {
            return layer->Forward({bottom1}, {top.get()});
        }
        return (Dtype) 0;
    }

    shared_ptr<Layer<Dtype>> layer;
    shared_ptr<Blob<Dtype>> top, bottom, checksum;
    Blob<Dtype> *bottom1;
    int type;
};

class ErrorRecord {
public:

    ErrorRecord(int n_, int m_, int k_, long idx_, double delta_) : n(n_), m(m_), k(k_), idx(idx_),
                                                                    delta(delta_) {}

    int n, m, k;
    long idx;
    double delta;

};

enum OperationEnum {
    D_Cw1,
    D_Cw2,
    Cd1_W,
    Cd2_W,
    Cd1_Cw1,
    Cd1_Cw2,
    Cd2_Cw1,
    Cd2_Cw2,
    Bias
};

std::vector<std::vector<double>> rerror;
std::vector<std::vector<double>> rerror_max;
bool profiling_rerror = false;
int num_rerror = 3;

std::map<string, std::vector<long>> correction_count;
vector<int> layer_abft_op;
vector<vector<double>> layer_abft_op_time;

int inject_one_fault(float *A, int error_bits) {
    if (error_bits <= 0) {
        return 0;
    } else if (error_bits > 32) {
        long r = rand();
//        (*A) += (r % 1000 + 1) * 1e19;
        (*A) += 1 + r % 100000;
        return 0;
    }

    float val = *A;
    unsigned char *c = reinterpret_cast<unsigned char *>(&val);

    int bit = (31 - error_bits - 1) + rand() % error_bits + 1;
    c[bit / 8] ^= 1UL << (bit % 8);
    *A = val;
    return bit;
}

template<typename Dtype>
inline bool has_error(Dtype sum, Dtype checksum, long layer, int op) {
    if (profiling_rerror) {
        if (fabs(sum - checksum) > rerror_max[layer][op]) {
            rerror_max[layer][op] = fabs(sum - checksum);
        }
        return false;
    }
    return (fabs(sum - checksum) > rerror[layer][op]);
}

template<typename Dtype>
inline bool not_equal(Dtype a, Dtype b) {
    return !(a / b > 0.90 && a / b < 1.11);
}


void layer_tuning(
        const vector<shared_ptr<Layer<float>>> &layers,
        const vector<vector<Blob<float> *>> &bottom_vecs,
        const vector<vector<Blob<float> *>> &top_vecs,
        std::map<string, std::map<int, shared_ptr<Operation<float>>>> &convOps) {
    for (int op = 1; op <= 3; op++)
        for (long l = 0; l < layers.size(); l++) {
            if (layers[l]->layer_param().type() != "Convolution") {
                layers[l]->Forward(bottom_vecs[l], top_vecs[l]);
                continue;
            }
            auto O = top_vecs[l][0];
            auto D = bottom_vecs[l][0];
            auto W = layers[l]->blobs()[0];
            auto N = O->shape(0);
            auto M = O->shape(1);
            auto K = O->count(2);
            auto ops = convOps[layers[l]->layer_param().name()];
            auto Cd1_blob = ops[Cd1_Cw1]->bottom;
            auto Cd2_blob = ops[Cd2_Cw1]->bottom;

            auto Co1 = ops[Cd1_W]->top;
            auto Co2 = ops[D_Cw1]->top;
            auto Co3 = ops[Cd2_W]->top;
            auto Co4 = ops[D_Cw2]->top;
            auto Co5 = ops[Cd1_Cw1]->top;
            auto Co6 = ops[Cd1_Cw2]->top;
            auto Co7 = ops[Cd2_Cw1]->top;
            auto Co8 = ops[Cd2_Cw2]->top;
            auto CCo1 = ops[Cd1_W]->checksum;
            auto CCo2 = ops[D_Cw1]->checksum;
            auto CCo3 = ops[Cd2_W]->checksum;
            auto CCo4 = ops[D_Cw2]->checksum;
            auto CCo5 = ops[Cd1_Cw1]->checksum;
            auto CCo6 = ops[Cd1_Cw2]->checksum;
            auto CCo7 = ops[Cd2_Cw1]->checksum;
            auto CCo8 = ops[Cd2_Cw2]->checksum;

            compute_checksum_column_block(D->shape(0), D->count(1),
                                          D->cpu_data(), 0,
                                          Cd1_blob->mutable_cpu_data(), 0);
            layers[l]->Forward(bottom_vecs[l], top_vecs[l]);

            ops[Cd1_Cw1]->Forward();
            compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                          CCo1->mutable_cpu_data(), 0);
            compute_checksum_column_block(M, K, CCo1->cpu_data(), 0,
                                          CCo5->mutable_cpu_data(), 0);
            compute_checksum_column_block(D->shape(0), D->count(1),
                                          D->cpu_data(), 0,
                                          Cd2_blob->mutable_cpu_data(), 1);
            ops[Cd1_Cw2]->Forward();
            ops[Cd2_Cw1]->Forward();
            compute_checksum_column_block(M, K, CCo1->cpu_data(), 0,
                                          CCo6->mutable_cpu_data(), 1);
            compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                          CCo3->mutable_cpu_data(), 1);
            compute_checksum_column_block(M, K, CCo3->cpu_data(), 0,
                                          CCo7->mutable_cpu_data(), 0);
            Timer timer;

            if (op == 1) {
                timer.Start();
                ops[D_Cw1]->Forward();
                ops[Cd1_W]->Forward();

                compute_checksum_row_block(N, M, K, O->cpu_data(), 0,
                                           CCo2->mutable_cpu_data(), 0);
                layer_abft_op_time[l][0] += timer.MicroSeconds();
            }
            if (op == 2) {
                timer.Start();
                ops[Cd1_W]->Forward();
                ops[Cd2_W]->Forward();

                compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                              CCo3->mutable_cpu_data(), 1);
                auto t1 = timer.MicroSeconds();
                layer_abft_op_time[l][1] += t1;

                timer.Start();
                ops[D_Cw1]->Forward();
                compute_checksum_row_block(N, M, K, O->cpu_data(), 0,
                                           CCo2->mutable_cpu_data(), 0);
                layer_abft_op_time[l][2] += t1 + timer.MicroSeconds();
            }
            if (op == 3) {
                timer.Start();
                ops[D_Cw1]->Forward();
                ops[D_Cw2]->Forward();
                compute_checksum_row_block(N, M, K, O->cpu_data(), 0,
                                           CCo2->mutable_cpu_data(), 0);
                compute_checksum_row_block(N, M, K, O->cpu_data(), 0,
                                           CCo4->mutable_cpu_data(), 1);
                auto t1 = timer.MicroSeconds();
                layer_abft_op_time[l][3] += t1;

                timer.Start();
                ops[Cd1_W]->Forward();
                layer_abft_op_time[l][4] += t1 + timer.MicroSeconds();
            }
        }
}

int inference_layer(
        const int l, const vector<shared_ptr<Layer<float>>> &layers,
        const vector<vector<Blob<float> *>> &bottom_vecs,
        const vector<vector<Blob<float> *>> &top_vecs,
        std::map<string, std::map<int, shared_ptr<Operation<float>>>> &convOps,
        int abft_conv_op, std::vector<double> &overheads, double &overhead,
        int inject_layer) {
    string layer_name = layers[l]->layer_param().name();
    string profilingLayerStr = "Profiling Layer," + layer_name + ",";
    LOG(INFO) << profilingLayerStr << layers[l]->layer_param().type();

    if (layers[l]->layer_param().type() != "Convolution" || abft_conv_op == 0) {
        layers[l]->Forward(bottom_vecs[l], top_vecs[l]);
        return 0;
    }

    if (correction_count.count(layer_name) == 0) {
        correction_count[layer_name] = std::vector<long>(5, 0);
    }
    auto O = top_vecs[l][0];
    auto D = bottom_vecs[l][0];
    auto W = layers[l]->blobs()[0];
    auto N = O->shape(0);
    auto M = O->shape(1);
    auto K = O->count(2);

    Timer timer;
    timer.Start();

    auto ops = convOps[layers[l]->layer_param().name()];
    auto Cd1_blob = ops[Cd1_Cw1]->bottom;
    auto Cd2_blob = ops[Cd2_Cw1]->bottom;

    compute_checksum_column_block(D->shape(0), D->count(1), D->cpu_data(), 0,
                                  Cd1_blob->mutable_cpu_data(), 0);

    overheads[0] += timer.MicroSeconds();
    LOG(INFO) << profilingLayerStr
              << "Encode Cd1 Time: " << timer.MicroSeconds() / 1000;

    timer.Start();
    layers[l]->Forward(bottom_vecs[l], top_vecs[l]);
    LOG(INFO) << profilingLayerStr
              << "Forward Time: " << timer.MicroSeconds() / 1000;

    std::map<long, double> inject_records;
    std::vector<ErrorRecord> fixed_records;

    if (inject_layer == l) {
        int inject_bits = caffe_getenv("FT_ERROR_BITS");
        int inject_op = caffe_getenv("FT_ERROR");

        // inject_op: 101 -> single block; 102 -> single row; 103 -> single column; 104-> row/column
        if (inject_op == 104) {
            long input_weight = D->count(0);
            long kernel_weight = W->count(0);
            int t = rand() % (input_weight + kernel_weight);
            if (t < input_weight) {
                inject_op = 102;
            } else {
                inject_op = 103;
            }
        }

        if (inject_bits == 0) {
            inject_bits = inject_op;
        }

        float *data_p = O->mutable_cpu_data();
        long injected_n = rand() % N;
        long injected_m = rand() % M;

        std::vector<int> injectlist_k(1 + rand() % 10);
        for (auto &injected_k: injectlist_k) {
            injected_k = rand() % K;
        }

        for (int i = 0; i < (inject_op == 101 ? 1 : 2 + rand() % 10); i++) {
            if (inject_op == 102) {
                injected_m = rand() % M;
            } else if (inject_op == 103) {
                injected_n = rand() % N;
            }
            for (auto &injected_k:injectlist_k) {
                long idx = injected_n * M * K + injected_m * K + injected_k;

                if (inject_records.find(idx) != inject_records.end()) {
                    continue;
                }
                float oldVal = data_p[idx];
                inject_records[idx] = oldVal;

                int injected_bit =
                        inject_one_fault(data_p + idx, inject_bits);
                if (injected_bit != -1) {
                    LOG(WARNING)
                            << "Inject Fault, "
                            << "Layer: " << layers[l]->layer_param().name()
                            << " top[" << idx << "][" << injected_n << ","
                            << injected_m << "," << injected_k << "] From "
                            << oldVal << " To "
                            << O->mutable_cpu_data()[idx]
                            << ", bit position=" << injected_bit;
                }
            }
        }
        LOG(WARNING) << "Inject Fault, Total:" << inject_records.size();
    }

    auto Co1 = ops[Cd1_W]->top;
    auto Co2 = ops[D_Cw1]->top;
    auto Co3 = ops[Cd2_W]->top;
    auto Co4 = ops[D_Cw2]->top;
    auto Co5 = ops[Cd1_Cw1]->top;
    auto Co6 = ops[Cd1_Cw2]->top;
    auto Co7 = ops[Cd2_Cw1]->top;
    auto Co8 = ops[Cd2_Cw2]->top;
    auto CCo1 = ops[Cd1_W]->checksum;
    auto CCo2 = ops[D_Cw1]->checksum;
    auto CCo3 = ops[Cd2_W]->checksum;
    auto CCo4 = ops[D_Cw2]->checksum;
    auto CCo5 = ops[Cd1_Cw1]->checksum;
    auto CCo6 = ops[Cd1_Cw2]->checksum;
    auto CCo7 = ops[Cd2_Cw1]->checksum;
    auto CCo8 = ops[Cd2_Cw2]->checksum;

    timer.Start();

    double bias_sum = 0;
    double bias_msum = 0;
    vector<double> bias_n(M, 0);
    vector<double> bias_n2(M, 0);
    if (layers[l]->layer_param().convolution_param().bias_term()) {
        auto bias_ptr = layers[l]->blobs()[1]->cpu_data();
        bias_sum = BlobSum(layers[l]->blobs()[1].get());
        for (int m = 0; m < M; m++) {
            bias_msum += bias_ptr[m] * m;
            bias_n[m] = bias_ptr[m] * N;
            bias_n2[m] = bias_ptr[m] * (N * (N + 1) / 2);
        }
    }

    bool D_Cw1_done = false;
    bool Cd1_W_done = false;
    bool CCo1_done = false;
    bool CCo2_done = false;
    bool CCo3_done = false;
    bool Cd2_done = false;
    bool further_action = false;

    {
        ops[Cd1_Cw1]->Forward();

//        bottleneck is to compute CCo5, 2.854s vs ops[Cd1_Cw1]->Forward (0.115s)
//        compute_checksum_column_block(N * M, K, O->cpu_data(), 0, CCo5->mutable_cpu_data(), 0);

        compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                      CCo1->mutable_cpu_data(), 0);
        CCo1_done = true;

        compute_checksum_column_block(M, K, CCo1->cpu_data(), 0,
                                      CCo5->mutable_cpu_data(), 0);

        auto sum_ptr = CCo5->cpu_data();
        auto ck_ptr = Co5->cpu_data();

#ifdef _OPENMP
#pragma omp parallel for shared(further_action)
#endif
        for (int k = 0; k < K; k++) {
            if (further_action) {
                continue;
            }
            double sum = sum_ptr[k] - bias_sum * N;
            double ck = ck_ptr[k];
            if (has_error(sum, ck, l, 2)) {
                LOG(WARNING) << profilingLayerStr << "Error detected by CoC-D! k=" << k << ", Sum=" << sum << ", CK="
                             << ck;
                further_action = true;
            }
        }
    }

    {
        auto t = timer.MicroSeconds();
        overheads[0] += t;
        overhead += t;
        LOG(INFO) << profilingLayerStr
                  << "Detect Time: " << timer.MicroSeconds() / 1000;
    }


    if (!further_action && !profiling_rerror) {
        return 0;
    }

/******************* Start to locate and correct error  ****************/
    correction_count[layer_name][0]++;

    timer.Start();

    further_action = false;
    fixed_records.clear();
    {
        if (!Cd2_done) {
            compute_checksum_column_block(D->shape(0), D->count(1),
                                          D->cpu_data(), 0,
                                          Cd2_blob->mutable_cpu_data(), 1);
        }
        ops[Cd1_Cw2]->Forward();
        ops[Cd2_Cw1]->Forward();
        compute_checksum_column_block(M, K, CCo1->cpu_data(), 0,
                                      CCo6->mutable_cpu_data(), 1);
        if (!CCo3_done) {
            compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                          CCo3->mutable_cpu_data(), 1);
            CCo3_done = true;
        }
        compute_checksum_column_block(M, K, CCo3->cpu_data(), 0,
                                      CCo7->mutable_cpu_data(), 0);

        auto sum_ptr = CCo5->cpu_data();
        auto sum2_ptr = CCo6->cpu_data();
        auto sum3_ptr = CCo7->cpu_data();
        auto ck_ptr = Co5->cpu_data();
        auto ck2_ptr = Co6->cpu_data();
        auto ck3_ptr = Co7->cpu_data();

#ifdef _OPENMP
#pragma omp parallel for shared(further_action)
#endif
        for (int k = 0; k < K; k++) {
            if (further_action) {
                continue;
            }
            double sum = sum_ptr[k] - bias_sum * N;
            double ck = ck_ptr[k];
            if (has_error(sum, ck, l, 2)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    double delta = ck - sum;
                    double mm0 = (ck2_ptr[k] - (sum2_ptr[k] - bias_msum * N)) / delta - 1;
                    double nn0 = (ck3_ptr[k] - (sum3_ptr[k] - bias_sum * ((N + 1) * N / 2))) / delta - 1;
                    int mm = round(mm0);
                    int nn = round(nn0);
                    if (fabs(mm - mm0) > 0.4 || fabs(nn - nn0) > 0.4 || nn < 0 || mm < 0 || nn >= N || mm >= M) {
                        LOG(WARNING) << profilingLayerStr
                                     << "Error couldn't be fixed by CoC, k=" << k << ", delta= " << delta << ", m="
                                     << mm0 << ", n=" << nn0 << "; sum=" << sum3_ptr[k] << ", ck=" << ck3_ptr[k]
                                     << ", bias=" << bias_sum * ((M + 1) * M / 2);
                        further_action = true;
                    } else {
                        long idx = k + mm * K + nn * M * K;
                        auto er = ErrorRecord(nn, mm, k, idx, delta);
                        fixed_records.push_back(er);
                        LOG(WARNING) << profilingLayerStr
                                     << "Error could be fixed by CoC, position ["
                                     << k + mm * K + nn * M * K << "][" << nn << ","
                                     << mm << "," << k << "], delta = " << delta;
                    }
                }
            }
        }

        auto t = timer.MicroSeconds();
        overheads[1] += t;
        overhead += t;
        LOG(INFO) << profilingLayerStr
                  << "CoC Time: " << timer.MicroSeconds() / 1000;
    }

    if (!further_action && !profiling_rerror) {
        bool verify = true;

        if (!Cd1_W_done) {
            ops[Cd1_W]->Forward();
            Cd1_W_done = true;
        }
        if (!D_Cw1_done) {
            ops[D_Cw1]->Forward();
            D_Cw1_done = true;
        }
        auto O_ptr = O->mutable_cpu_data();
        auto Co1_ptr = Co1->cpu_data();
        auto Co2_ptr = Co2->cpu_data();
        std::vector<bool> checked_bycolumn(fixed_records.size(), false);
        std::vector<bool> checked_byrow(fixed_records.size(), false);
        for (int i = 0; i < fixed_records.size(); i++) {
            if (!checked_bycolumn[i]) {
                checked_bycolumn[i] = true;
                auto &record = fixed_records[i];
                double sum_column = 0, sum_delta = record.delta;
                for (int nn = 0; nn < N; nn++) {
                    sum_column += O_ptr[nn * M * K + record.m * K + record.k];
                }
                for (int j = i + 1; j < fixed_records.size(); j++) {
                    if (!checked_bycolumn[j] && fixed_records[j].m == record.m && fixed_records[j].k == record.k) {
                        sum_delta += fixed_records[j].delta;
                        checked_bycolumn[j] = true;
                    }
                }
                if (not_equal<double>(sum_column + sum_delta, Co1_ptr[record.m * K + record.k])) {
                    verify = false;
                    LOG(WARNING) << profilingLayerStr
                                 << "Verification Error, error couldn't be fixed by CoC, position ["
                                 << record.k + record.m * K + record.n * M * K << "][" << record.n << ","
                                 << record.m << "," << record.k << "], sum=" << sum_column << ", ck="
                                 << Co1_ptr[record.m * K + record.k] << ", delta="
                                 << sum_delta;
                    break;
                }
            }
            if (!checked_byrow[i]) {
                checked_byrow[i] = true;
                auto &record = fixed_records[i];
                double sum_row = 0, sum_delta = record.delta;
                for (int mm = 0; mm < M; mm++) {
                    sum_row += O_ptr[record.n * M * K + mm * K + record.k];
                }
                for (int j = i + 1; j < fixed_records.size(); j++) {
                    if (!checked_byrow[j] && fixed_records[j].n == record.n && fixed_records[j].k == record.k) {
                        sum_delta += fixed_records[j].delta;
                        checked_byrow[j] = true;
                    }
                }
                if (not_equal<double>(sum_row + sum_delta, Co2_ptr[record.n * K + record.k])) {
                    verify = false;
                    LOG(WARNING) << profilingLayerStr
                                 << "Verification Error, error couldn't be fixed by CoC, position ["
                                 << record.k + record.m * K + record.n * M * K << "][" << record.n << ","
                                 << record.m << "," << record.k << "], sum=" << sum_row << ", ck="
                                 << Co2_ptr[record.n * K + record.k] << ", delta="
                                 << sum_delta;
                    break;
                }
            }
        }

        if (verify) {
            for (auto &record:fixed_records) {
                O_ptr[record.idx] += record.delta;
                LOG(WARNING) << profilingLayerStr
                             << "Error Fixed by CoC, position ["
                             << record.k + record.m * K + record.n * M * K << "][" << record.n << ","
                             << record.m << "," << record.k << "]";
            }
            correction_count[layer_name][1]++;
            return 1;
        }
    }

    timer.Start();


    bool use_input_checksum =
            (abft_conv_op == 12 || (abft_conv_op == 14 && layer_abft_op[l] == 2) || profiling_rerror);
    further_action = !use_input_checksum;
    fixed_records.clear();
    if (use_input_checksum) {
        timer.Start();
        if (!Cd2_done) {
            compute_checksum_column_block(D->shape(0), D->count(1), D->cpu_data(),
                                          0, Cd2_blob->mutable_cpu_data(), 1);
        }
        if (!Cd1_W_done) {
            ops[Cd1_W]->Forward();
            Cd1_W_done = true;
        }
        ops[Cd2_W]->Forward();

        if (!CCo1_done) {
            compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                          CCo1->mutable_cpu_data(), 0);
            CCo1_done = true;
        }

        if (!CCo3_done) {
            compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                          CCo3->mutable_cpu_data(), 1);
            CCo3_done = true;
        }

        auto sum_ptr = CCo1->cpu_data();
        auto sum2_ptr = CCo3->cpu_data();
        auto ck_ptr = Co1->cpu_data();
        auto ck2_ptr = Co3->cpu_data();

#ifdef _OPENMP
#pragma omp parallel for shared(further_action)
#endif
        for (int m = 0; m < M; m++) {
            if (further_action) {
                continue;
            }
            for (int k = 0; !further_action && k < K; k++) {
                double sum = sum_ptr[k + m * K] - bias_n[m];
                double ck = ck_ptr[k + m * K];
                if (has_error(sum, ck, l, 0)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    {
                        double sum2 = sum2_ptr[k + m * K] - bias_n2[m];
                        double ck2 = ck2_ptr[k + m * K];
                        double nn0 = fabs((sum2 - ck2) / (sum - ck)) - 1;
                        int nn = round(nn0);

                        if (fabs(nn - nn0) > 0.4 || nn < 0 || nn >= N) {
                            LOG(WARNING) << profilingLayerStr
                                         << "Error could't be fixed by RC, n:" << nn0 << "; m:" << m << "; k:" << k;
                            further_action = true;
                        } else {
                            long idx = k + m * K + nn * M * K;
                            auto er = ErrorRecord(nn, m, k, idx, ck - sum);
                            fixed_records.push_back(er);
                            LOG(WARNING) << profilingLayerStr
                                         << "Error could be fixed by RC, position ["
                                         << nn << "," << m << "," << k << "]("
                                         << k + m * K + nn * M * K << "), delta= "
                                         << ck - sum;
                        }
                    }
                }
            }
        }
        auto t = timer.MicroSeconds();
        overheads[2] += t;
        overhead += t;
        LOG(INFO) << profilingLayerStr
                  << "RC Time: " << timer.MicroSeconds() / 1000;
    }
    if (!further_action && !profiling_rerror) {
        bool verify = true;

        if (!D_Cw1_done) {
            ops[D_Cw1]->Forward();
            D_Cw1_done = true;
        }
        auto O_ptr = O->mutable_cpu_data();
        auto Co2_ptr = Co2->cpu_data();
        std::vector<bool> checked_byrow(fixed_records.size(), false);
        for (int i = 0; i < fixed_records.size(); i++) {
            if (!checked_byrow[i]) {
                checked_byrow[i] = true;
                auto &record = fixed_records[i];
                double sum_row = 0, sum_delta = record.delta;
                for (int mm = 0; mm < M; mm++) {
                    sum_row += O_ptr[record.n * M * K + mm * K + record.k];
                }
                for (int j = i + 1; j < fixed_records.size(); j++) {
                    if (!checked_byrow[j] && fixed_records[j].n == record.n && fixed_records[j].k == record.k) {
                        sum_delta += fixed_records[j].delta;
                        checked_byrow[j] = true;
                    }
                }
                if (not_equal<double>(sum_row + sum_delta, Co2_ptr[record.n * K + record.k])) {
                    verify = false;
                    LOG(WARNING) << profilingLayerStr
                                 << "Verification Error, error couldn't be fixed by RC, position ["
                                 << record.k + record.m * K + record.n * M * K << "][" << record.n << ","
                                 << record.m << "," << record.k << "], sum=" << sum_row << ", ck="
                                 << Co2_ptr[record.n * K + record.k] << ", delta="
                                 << sum_delta;
                    break;
                }
            }
        }

        if (verify) {
            for (auto &record:fixed_records) {
                O_ptr[record.idx] += record.delta;
                LOG(WARNING) << profilingLayerStr
                             << "Error Fixed by RC, position ["
                             << record.k + record.m * K + record.n * M * K << "][" << record.n << ","
                             << record.m << "," << record.k << "]";
            }
            correction_count[layer_name][2]++;
            return 1;
        }
    }

    bool use_kernel_checksum =
            (abft_conv_op == 13 || (abft_conv_op == 14 && layer_abft_op[l] == 3) || profiling_rerror);
    further_action = !use_kernel_checksum;
    fixed_records.clear();
    if (use_kernel_checksum) {
        timer.Start();
        if (!D_Cw1_done) {
            ops[D_Cw1]->Forward();
            D_Cw1_done = true;
        }
        ops[D_Cw2]->Forward();
        if (!CCo2_done) {
            compute_checksum_row_block(N, M, K, O->cpu_data(), 0,
                                       CCo2->mutable_cpu_data(), 0);
            CCo2_done = true;
        }
        compute_checksum_row_block(N, M, K, O->cpu_data(), 0,
                                   CCo4->mutable_cpu_data(), 1);

        auto sum_ptr = CCo2->cpu_data();
        auto sum2_ptr = CCo4->cpu_data();
        auto ck_ptr = Co2->cpu_data();
        auto ck2_ptr = Co4->cpu_data();

#ifdef _OPENMP
#pragma omp parallel for shared(further_action)
#endif
        for (int n = 0; n < N; n++) {
            if (further_action) {
                continue;
            }
            for (int k = 0; !further_action && k < K; k++) {
                double sum = sum_ptr[k + n * K] - bias_sum;
                double ck = ck_ptr[k + n * K];
                if (has_error(sum, ck, l, 1)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    {
                        double sum2 = sum2_ptr[k + n * K] - bias_msum;
                        double ck2 = ck2_ptr[k + n * K];
                        double mm0 = fabs((sum2 - ck2) / (sum - ck)) - 1;
                        int mm = round(mm0);
                        if (fabs(mm - mm0) > 0.4 || mm < 0 || mm >= M) {
                            LOG(WARNING) << profilingLayerStr
                                         << "Error could't be fixed by ClC, n:" << n << "; m:" << mm0 << "; k:" << k;
                            further_action = true;
                        } else {
                            long idx = k + mm * K + n * M * K;
                            auto er = ErrorRecord(n, mm, k, idx, ck - sum);
                            fixed_records.push_back(er);
                            LOG(WARNING) << profilingLayerStr
                                         << "Error could be fixed by ClC, position [" << k + mm * K + n * M * K
                                         << "], delta= " << ck - sum;
                        }
                    }
                }
            }
        }
        auto t = timer.MicroSeconds();
        overheads[3] += t;
        overhead += t;
        LOG(INFO) << profilingLayerStr
                  << "ClC Time: " << timer.MicroSeconds() / 1000;
    }

    if (!further_action && !profiling_rerror) {
        bool verify = true;
        if (!Cd1_W_done) {
            ops[Cd1_W]->Forward();
            Cd1_W_done = true;
        }
        auto O_ptr = O->mutable_cpu_data();
        auto Co1_ptr = Co1->cpu_data();
        std::vector<bool> checked_bycolumn(fixed_records.size(), false);
        for (int i = 0; i < fixed_records.size(); i++) {
            if (!checked_bycolumn[i]) {
                checked_bycolumn[i] = true;
                auto &record = fixed_records[i];
                double sum_column = 0, sum_delta = record.delta;
                for (int nn = 0; nn < N; nn++) {
                    sum_column += O_ptr[nn * M * K + record.m * K + record.k];
                }
                for (int j = i + 1; j < fixed_records.size(); j++) {
                    if (!checked_bycolumn[j] && fixed_records[j].m == record.m && fixed_records[j].k == record.k) {
                        sum_delta += fixed_records[j].delta;
                        checked_bycolumn[j] = true;
                    }
                }
                if (not_equal<double>(sum_column + sum_delta, Co1_ptr[record.m * K + record.k])) {
                    verify = false;
                    LOG(WARNING) << profilingLayerStr
                                 << "Verification Error, error couldn't be fixed by ClC, position ["
                                 << record.k + record.m * K + record.n * M * K << "][" << record.n << ","
                                 << record.m << "," << record.k << "], sum=" << sum_column << ", ck="
                                 << Co1_ptr[record.m * K + record.k] << ", delta="
                                 << sum_delta;
                    break;
                }
            }
        }

        if (verify) {
            for (auto &record:fixed_records) {
                O_ptr[record.idx] += record.delta;
                LOG(WARNING) << profilingLayerStr
                             << "Error Fixed by ClC, position ["
                             << record.k + record.m * K + record.n * M * K << "][" << record.n << ","
                             << record.m << "," << record.k << "]";
            }
            correction_count[layer_name][3]++;
            return 1;
        }
    }

    further_action = false;
    fixed_records.clear();
    {
        timer.Start();
        if (!D_Cw1_done) {
            ops[D_Cw1]->Forward();
            D_Cw1_done = true;
        }
        if (!Cd1_W_done) {
            ops[Cd1_W]->Forward();
            Cd1_W_done = true;
        }

        if (!CCo1_done) {
            compute_checksum_column_block(N, M * K, O->cpu_data(), 0,
                                          CCo1->mutable_cpu_data(), 0);
            CCo1_done = true;
        }

        if (!CCo2_done) {
            compute_checksum_row_block(N, M, K, O->cpu_data(), 0,
                                       CCo2->mutable_cpu_data(), 0);
            CCo2_done = true;
        }

        vector<vector<int>> sdc_n(K, vector<int>());
        vector<vector<int>> sdc_m(K, vector<int>());
        vector<vector<double>> sdc_delta_n(K, vector<double>());
        vector<vector<double>> sdc_delta_m(K, vector<double>());
        auto sum_ptr = CCo1->cpu_data();
        auto ck_ptr = Co1->cpu_data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                double sum = sum_ptr[k + m * K] - bias_n[m];
                double ck = ck_ptr[k + m * K];
                if (has_error(sum, ck, l, 0)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    {
                        LOG(WARNING) << profilingLayerStr << "Error detected by FC, sum-ck: "
                                     << sum - ck << ", eb=" << rerror[l][0];
                        sdc_m[k].push_back(m);
                        sdc_delta_m[k].push_back(ck - sum);
                    }
                }
            }
        }

        sum_ptr = CCo2->cpu_data();
        ck_ptr = Co2->cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                double sum = sum_ptr[k + n * K] - bias_sum;
                double ck = ck_ptr[k + n * K];
                if (has_error(sum, ck, l, 1)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    {
                        LOG(WARNING) << profilingLayerStr << "Error detected by FC, sum-ck="
                                     << sum - ck << ", eb=" << rerror[l][1];
                        sdc_n[k].push_back(n);
                        sdc_delta_n[k].push_back(ck - sum);
                    }
                }
            }
        }

        for (int k = 0; !further_action && k < K; k++) {
            if (sdc_n[k].size() == 0 && sdc_m[k].size() == 0) {
                continue;
            }

            if (sdc_n[k].size() == 1 && sdc_m[k].size() == 1) {
                double delta_n = sdc_delta_n[k][0];
                double delta_m = sdc_delta_m[k][0];

                bool equal = ((delta_n - delta_m) / (delta_n + delta_m) < 0.2);
                if (equal) {
                    long idx = k + sdc_m[k][0] * K + sdc_n[k][0] * M * K;
                    auto er = ErrorRecord(sdc_n[k][0], sdc_m[k][0], k, idx, (delta_n + delta_m) / 2);
                    fixed_records.push_back(er);
                    LOG(WARNING) << profilingLayerStr << "Error Fixed by FC, position ["
                                 << k + sdc_m[k][0] * K + sdc_n[k][0] * M * K << "], delta="
                                 << (delta_n + delta_m) / 2;
                } else {
                    LOG(ERROR) << profilingLayerStr << "Error cannot be fixed by FC, position ["
                               << k + sdc_m[k][0] * K + sdc_n[k][0] * M * K
                               << "], sdc_delta_n=" << delta_n
                               << ", sdc_delta_m = " << delta_m;
                    further_action = true;
                    break;
                }
            } else if (sdc_n[k].size() > 1 && sdc_m[k].size() == 1) {
                double delta_m = sdc_delta_m[k][0];
                double delta_n = 0;
                for (auto &tmp : sdc_delta_n[k]) {
                    delta_n += tmp;
                }

                bool equal = ((delta_n - delta_m) / (delta_n + delta_m) < 0.2);
                if (equal) {
                    for (int p = 0; p < sdc_n[k].size(); p++) {
                        long idx = k + sdc_m[k][0] * K + sdc_n[k][p] * M * K;
                        auto er = ErrorRecord(sdc_n[k][p], sdc_m[k][0], k, idx, sdc_delta_n[k][p]);
                        fixed_records.push_back(er);
                        LOG(WARNING) << profilingLayerStr << "Error Fixed by FC, position ["
                                     << k + sdc_m[k][0] * K + sdc_n[k][p] * M * K
                                     << "], delta=" << sdc_delta_n[k][p];
                    }
                } else {
                    LOG(ERROR) << profilingLayerStr << "Error couldn't be fixed by FC, m = ["
                               << sdc_m[k][0] << "], sdc_delta_n=" << delta_n
                               << ", sdc_delta_m = " << delta_m;
                    further_action = true;
                    break;
                }
            } else if (sdc_n[k].size() == 1 && sdc_m[k].size() > 1) {
                double delta_n = sdc_delta_n[k][0];
                double delta_m = 0;
                for (auto &tmp : sdc_delta_m[k]) {
                    delta_m += tmp;
                }

                bool equal = ((delta_n - delta_m) / (delta_n + delta_m) < 0.2);
                if (equal) {
                    for (int p = 0; p < sdc_m[k].size(); p++) {
                        long idx = k + sdc_m[k][p] * K + sdc_n[k][0] * M * K;
                        auto er = ErrorRecord(sdc_n[k][0], sdc_m[k][p], k, idx, sdc_delta_m[k][p]);
                        fixed_records.push_back(er);
                        LOG(WARNING) << profilingLayerStr
                                     << "Error Fixed by FC, position ["
                                     << k + sdc_m[k][p] * K + sdc_n[k][0] * M * K
                                     << "], delta=" << sdc_delta_m[k][p];
                    }
                } else {
                    LOG(ERROR) << profilingLayerStr
                               << "Error couldn't be fixed by FC, n = ["
                               << sdc_n[k][0] << "], sdc_delta_n=" << delta_n
                               << ", sdc_delta_m = " << delta_m;
                    further_action = true;
                    break;
                }

            } else {
                LOG(WARNING) << profilingLayerStr << "Error couldn't be fixed by FC, need restart. k =" << k;
                further_action = true;
                break;
            }
        }
        auto t = timer.MicroSeconds();
        overheads[4] += t;
        overhead += t;
        LOG(INFO) << profilingLayerStr
                  << "FC Time: " << timer.MicroSeconds() / 1000;
    }

    if (!further_action && !profiling_rerror) {
        for (auto &record:fixed_records) {
            O->mutable_cpu_data()[record.idx] += record.delta;
        }
        correction_count[layer_name][4]++;
        return 1;
    }
    if (profiling_rerror) {
        return 0;
    }
    LOG(ERROR) << "ERROR Cannot fix by ABFT, need restart! Total Error: " << inject_records.size() << ", Fixed:"
               << fixed_records.size();
    return 2;
}

int fault_detected = 0;
int fault_not_detected = 0;
int no_fault_false_detected = 0;

void inference_net(
        const vector<shared_ptr<Layer<float>>> &layers,
        const vector<vector<Blob<float> *>> &bottom_vecs,
        const vector<vector<Blob<float> *>> &top_vecs,
        std::map<string, std::map<int, shared_ptr<Operation<float>>>> &convOps,
        int abft_conv_op, std::vector<double> &forward_time_per_layer,
        std::map<std::string, std::vector<double>> &overhead_per_layer,
        std::map<std::string, double> &rerun_time_per_layer, int inject_layer) {
    Timer forward_timer;

    int layer = 0;  // layer index
    auto &overheads = overhead_per_layer[layers[layer]->layer_param().name()];
    bool rerun = false;
    while (layer < layers.size()) {
        double overhead = 0;
        bool inject_fault = rerun ? false : inject_layer == layer;

        forward_timer.Start();
        int result = inference_layer(layer, layers, bottom_vecs, top_vecs,
                                     convOps, abft_conv_op, overheads, overhead,
                                     rerun ? -1 : inject_layer);

        forward_time_per_layer[layer] += forward_timer.MicroSeconds();

        //        LOG(ERROR) << layers[layer]->layer_param().name() << "  " <<
        //        result;
        if (inject_fault) {
            if (result == 0) {
                fault_not_detected++;
            } else {
                fault_detected++;
            }
        } else if (result != 0) {
            no_fault_false_detected++;
        }

        if (result == 2) {
            rerun_time_per_layer[layers[layer]->layer_param().name()] +=
                    forward_timer.MicroSeconds();
            rerun = true;
        } else {
            overhead_per_layer[layers[layer]->layer_param().name()] = overheads;
            layer++;
            if (layer >= layers.size()) {
                break;
            }
            overheads = overhead_per_layer[layers[layer]->layer_param().name()];
            rerun = false;
        }
    }
}

void inference_net_test(
        const vector<shared_ptr<Layer<float>>> &layers,
        const vector<vector<Blob<float> *>> &bottom_vecs,
        const vector<vector<Blob<float> *>> &top_vecs,
        std::map<string, std::map<int, shared_ptr<Operation<float>>>> &convOps,
        int abft_conv_op) {
    std::vector<double> forward_time_per_layer(layers.size(), 0.0);
    std::map<std::string, std::vector<double>> overhead_per_layer;
    std::map<std::string, double> rerun_time_per_layer;
    for (int i = 0; i < layers.size(); i++) {
        std::vector<double> overheads(5, 0.0);
        overhead_per_layer[layers[i]->layer_param().name()] = overheads;
        rerun_time_per_layer[layers[i]->layer_param().name()] = 0.0;
    }
    inference_net(layers, bottom_vecs, top_vecs, convOps, abft_conv_op,
                  forward_time_per_layer, overhead_per_layer,
                  rerun_time_per_layer, -1);
}

void cvReadImg(int imageIdx, Blob<float> *input_layer, int inputIdx) {
#ifdef USE_OPENCV
    int channel = input_layer->shape(1);
    int height = input_layer->shape(2);
    int width = input_layer->shape(3);
    float *input_data =
        input_layer->mutable_cpu_data() + inputIdx * input_layer->count(1);

    std::stringstream idss;
    idss << std::setw(8) << std::setfill('0') << imageIdx + 1;
    std::string ids = idss.str();
    auto imgNames = FLAGS_val_dir + "/images/ILSVRC2012_val_" + ids + ".JPEG";

    cv::Mat img = cv::imread(imgNames, -1);
    CHECK(!img.empty()) << "Unable to read image file: " << imgNames;
    //    LOG(INFO) << "Read image file: " << imgNames;

    bool is_img_grayscale = img.channels() == 1;
    bool is_img_bgr = img.channels() == 3;
    bool is_img_bgra = img.channels() == 4;
    bool is_color = true;

    if (!(is_img_grayscale || is_img_bgr || is_img_bgra)) {
        LOG(ERROR) << "Images with " << img.channels()
                   << " channels unsupported: " << imgNames;
    }
    bool need_convert = is_img_bgra || (is_img_grayscale == is_color);
    if (need_convert) {
        int conv_code = (is_img_grayscale && is_color)
                            ? cv::COLOR_GRAY2BGR
                            : (is_img_bgr && !is_color)
                                  ? cv::COLOR_BGR2GRAY
                                  : (is_img_bgra && is_color)
                                        ? cv::COLOR_BGRA2BGR
                                        : (is_img_bgra && !is_color)
                                              ? cv::COLOR_BGRA2GRAY
                                              : -1;
        cv::cvtColor(img, img, conv_code);
    }

    cv::Mat img_resized;
    if (img.size() != cv::Size(width, height)) {
        cv::resize(img, img_resized, cv::Size(width, height));
    } else {
        img_resized = img;
    }
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32FC3);

    vector<cv::Mat> input_channels;
    for (int c = 0; c < channel; c++) {
        cv::Mat cvchannel(height, width, CV_32FC1,
                          input_data + c * width * height);
        input_channels.push_back(cvchannel);
    }
    cv::split(img_float, input_channels);
#else
    LOG(ERROR) << "OpenCV is required to read Img!";
#endif
}

void cvReadImg(int imageIdx, int batch_size, Blob<float> *input_layer) {
    return cvReadImg(imageIdx, input_layer, imageIdx % batch_size);
}

// Time: benchmark the execution time of a model.
int inference() {
    srand(time(0));
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
    caffe::Phase phase = caffe::TEST;
    vector<string> stages = get_stages_from_flags(FLAGS_stage);

    // Set device id and mode
    LOG(WARNING) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
    // Instantiate the caffe net.
    Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages, NULL,
                         FLAGS_engine);
    if (FLAGS_weights.size()) {
        caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
    }
    PERFORMANCE_INIT_MONITOR();
    // Do a clean forward and backward pass, so that memory allocation are done
    // and future iterations will be more stable.
    //    LOG(WARNING) << "Performing Forward";
    // Note that for the speed benchmark, we will assume that the network does
    // not take any input blobs.
    //    float initial_loss;
    //    caffe_net.Forward(&initial_loss);
    //    LOG(WARNING) << "Initial loss: " << initial_loss;

    const vector<shared_ptr<Layer<float>>> &layers = caffe_net.layers();
    const vector<vector<Blob<float> *>> &bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float> *>> &top_vecs = caffe_net.top_vecs();
    vector<shared_ptr<Blob<float>>> fc_top_ck_blobs;
    std::map<string, std::map<int, shared_ptr<Operation<float>>>> convOps;
    auto ABFT_CONV_OPTION = caffe_getenv("FT_CONV");
    profiling_rerror = caffe_getenv("FT_PF") != 0;
    int INJECT_ERROR = caffe_getenv("FT_ERROR");
    if (profiling_rerror) {
        INJECT_ERROR = 0;
    }

    std::vector<std::string> model_path;
    boost::algorithm::split(model_path, FLAGS_model, boost::is_any_of(",/-."));
    auto model_name = model_path[model_path.size() - 2];

    rerror.resize(layers.size());
    rerror_max.resize(layers.size());
    std::string rerror_str = FLAGS_weights.replace(
            FLAGS_weights.find("caffemodel"), sizeof("caffemodel") - 1, "txt");
    std::ifstream rerror_infile(rerror_str);
    for (int i = 0; i < layers.size(); i++) {
        rerror[i].resize(num_rerror);
        rerror_max[i].resize(num_rerror);
        if (layers[i]->layer_param().type() == "Convolution") {
            int line;
            float eb1, eb2, eb3;
            if (rerror_infile.is_open() &&
                rerror_infile >> line >> eb1 >> eb2 >> eb3) {
                rerror[i][0] = eb1 * 10;
                rerror[i][1] = eb2 * 10;
                rerror[i][2] = eb3 * 10;
            } else {
                rerror[i][0] = 10;
                rerror[i][1] = 10;
                rerror[i][2] = 10;
            }
        }
    }
    for (int i = 0; i < layers.size(); i++) {
        if (layers[i]->layer_param().type() == "Convolution") {
            LOG(INFO) << "ROUND_OFF_ERROR " << i << " ; " << rerror[i][0] << " "
                      << " " << rerror[i][1] << " " << rerror[i][2];
        }
    }

    // Warm up 5 iterations here, because the first several iteration times
    // have huge variance in some machines.
    int warmup_iterations = 5;
    for (int j = 0; j < warmup_iterations; ++j) {
        if (j == warmup_iterations - 1) PERFORMANCE_START_RESETTING_MONITOR();
        for (int i = 0; i < layers.size(); ++i) {
            layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
        }
    }

    for (long i = 0; i < layers.size(); i++) {
        auto &layer_param = layers[i]->layer_param();

        if (layer_param.type() == "Convolution" && ABFT_CONV_OPTION != 0) {
            LOG(INFO) << "Prepare Layer," << layer_param.name() << ","
                      << layer_param.type();

            shared_ptr<Blob<float>> kernel_blob = layers[i]->blobs()[0];
            Blob<float> *D_blob = bottom_vecs[i][0];
            Blob<float> *O_blob = top_vecs[i][0];

            int group_ = layer_param.convolution_param().group();
            int num_output_ = layer_param.convolution_param().num_output();
            int kernel_dim_ = kernel_blob->count(1);
            int weight_offset_ = kernel_blob->count() / group_;

            LOG(INFO) << layer_param.name() << " Kernel Group: " << group_
                      << "; Shape: " << kernel_blob->shape_string()
                      << "; kernel_dim: " << kernel_dim_
                      << "; asum: " << kernel_blob->asum_data()
                      << "; sq: " << kernel_blob->sumsq_data();

            LOG(INFO) << layer_param.name() << " Bottom shape "
                      << bottom_vecs[i][0]->shape_string() << "; Top shape "
                      << top_vecs[i][0]->shape_string();

            vector<int> Cw_shape(kernel_blob->shape());
            Cw_shape[0] = 1;
            Cw_shape[1] *= group_;
            shared_ptr<Blob<float>> Cw1_blob(new Blob<float>(Cw_shape));
            shared_ptr<Blob<float>> Cw2_blob(new Blob<float>(Cw_shape));

            vector<int> Cd_shape(D_blob->shape());
            Cd_shape[0] = 1;
            shared_ptr<Blob<float>> Cd1_blob(new Blob<float>(Cd_shape));
            shared_ptr<Blob<float>> Cd2_blob(new Blob<float>(Cd_shape));

            vector<int> Co13_shape(O_blob->shape());
            Co13_shape[0] = 1;

            vector<int> Co24_shape(O_blob->shape());
            Co24_shape[1] = 1;

            vector<int> Co5678_shape(O_blob->shape());
            Co5678_shape[0] = 1;
            Co5678_shape[1] = 1;

            //            layers[i]->blobs().push_back(Cw1_blob);
            for (int g = 0; g < group_; ++g) {
                compute_checksum_column_block(
                        num_output_ / group_, kernel_dim_,
                        kernel_blob->cpu_data() + g * weight_offset_, 0,
                        Cw1_blob->mutable_cpu_data() + g * kernel_dim_, 0);
            }
            for (int g = 0; g < group_; ++g) {
                compute_checksum_column_block(
                        num_output_ / group_, kernel_dim_,
                        kernel_blob->cpu_data() + g * weight_offset_, 0,
                        Cw2_blob->mutable_cpu_data() + g * kernel_dim_,
                        1 + g * num_output_ / group_);
            }

            LOG(INFO) << layer_param.name()
                      << " Cw1 Blob Shape:" << Cw1_blob->shape_string()
                      << "; kernel_dim_" << Cw1_blob->count(1)
                      << "; asum:" << Cw1_blob->asum_data()
                      << "; sq:" << Cw1_blob->sumsq_data();

            std::map<int, shared_ptr<Operation<float>>> ops;

            caffe::LayerParameter commonParam;
            commonParam.set_type("Convolution");
            commonParam.set_engine(layer_param.engine());
            caffe::ConvolutionParameter *convParam =
                    commonParam.mutable_convolution_param();
            (*convParam) = layer_param.convolution_param();
            convParam->set_bias_term(false);

            //************************* Cd1 x W **********************
            convParam->set_group(layer_param.convolution_param().group());
            convParam->set_num_output(
                    layer_param.convolution_param().num_output());

            commonParam.set_name(layer_param.name() + "_Cd1_W");
            shared_ptr<Operation<float>> Cd1W_Op(
                    new Operation<float>(commonParam, Cd1_blob, Co13_shape));
            CHECK_EQ(Cd1W_Op->layer->blobs()[0]->count(), kernel_blob->count())
                << Cd1W_Op->layer->layer_param().name()
                << "has wrong blob size";
            Cd1W_Op->layer->blobs()[0] = kernel_blob;
            ops[Cd1_W] = Cd1W_Op;

            //************************* Cd2 x W **********************
            convParam->set_group(layer_param.convolution_param().group());
            convParam->set_num_output(
                    layer_param.convolution_param().num_output());

            commonParam.set_name(layer_param.name() + "_Cd2_W");
            shared_ptr<Operation<float>> Cd2W_Op(
                    new Operation<float>(commonParam, Cd2_blob, Co13_shape));
            CHECK_EQ(Cd2W_Op->layer->blobs()[0]->count(), kernel_blob->count())
                << Cd2W_Op->layer->layer_param().name()
                << "has wrong blob size";
            Cd2W_Op->layer->blobs()[0] = kernel_blob;
            ops[Cd2_W] = Cd2W_Op;

            //************************* D x Cw1 **********************
            convParam->set_group(1);
            convParam->set_num_output(1);

            commonParam.set_name(layer_param.name() + "_D_Cw1");
            shared_ptr<Operation<float>> DCw1_Op(
                    new Operation<float>(commonParam, D_blob, Co24_shape));
            CHECK_EQ(DCw1_Op->layer->blobs()[0]->count(), Cw1_blob->count())
                << DCw1_Op->layer->layer_param().name()
                << "has wrong blob size";
            DCw1_Op->layer->blobs()[0] = Cw1_blob;
            ops[D_Cw1] = DCw1_Op;

            //************************* D x Cw1 **********************
            convParam->set_group(1);
            convParam->set_num_output(1);

            commonParam.set_name(layer_param.name() + "_D_Cw2");
            shared_ptr<Operation<float>> DCw2_Op(
                    new Operation<float>(commonParam, D_blob, Co24_shape));
            CHECK_EQ(DCw2_Op->layer->blobs()[0]->count(), Cw2_blob->count())
                << DCw2_Op->layer->layer_param().name()
                << "has wrong blob size";
            DCw2_Op->layer->blobs()[0] = Cw2_blob;
            ops[D_Cw2] = DCw2_Op;

            //************************* Cd1 x Cw1 **********************
            commonParam.set_name(layer_param.name() + "_Cd1_Cw1");
            shared_ptr<Operation<float>> Cd1Cw1_Op(
                    new Operation<float>(commonParam, Cd1_blob, Co5678_shape));
            CHECK_EQ(Cd1Cw1_Op->layer->blobs()[0]->count(), Cw1_blob->count())
                << Cd1Cw1_Op->layer->layer_param().name()
                << "has wrong blob size";
            Cd1Cw1_Op->layer->blobs()[0] = Cw1_blob;
            ops[Cd1_Cw1] = Cd1Cw1_Op;

            //************************* Cd1 x Cw2 **********************
            commonParam.set_name(layer_param.name() + "_Cd1_Cw2");
            shared_ptr<Operation<float>> Cd1Cw2_Op(
                    new Operation<float>(commonParam, Cd1_blob, Co5678_shape));
            CHECK_EQ(Cd1Cw2_Op->layer->blobs()[0]->count(), Cw2_blob->count())
                << Cd1Cw2_Op->layer->layer_param().name()
                << "has wrong blob size";
            Cd1Cw2_Op->layer->blobs()[0] = Cw2_blob;
            ops[Cd1_Cw2] = Cd1Cw2_Op;

            //************************* Cd2 x Cw1 **********************
            commonParam.set_name(layer_param.name() + "_Cd2_Cw1");
            shared_ptr<Operation<float>> Cd2Cw1_Op(
                    new Operation<float>(commonParam, Cd2_blob, Co5678_shape));
            CHECK_EQ(Cd2Cw1_Op->layer->blobs()[0]->count(), Cw1_blob->count())
                << Cd2Cw1_Op->layer->layer_param().name()
                << "has wrong blob size";
            Cd2Cw1_Op->layer->blobs()[0] = Cw1_blob;
            ops[Cd2_Cw1] = Cd2Cw1_Op;

            //************************* Cd2 x Cw2 **********************
            commonParam.set_name(layer_param.name() + "_Cd2_Cw2");
            shared_ptr<Operation<float>> Cd2Cw2_Op(
                    new Operation<float>(commonParam, Cd2_blob, Co5678_shape));
            CHECK_EQ(Cd2Cw2_Op->layer->blobs()[0]->count(), Cw2_blob->count())
                << Cd2Cw2_Op->layer->layer_param().name()
                << "has wrong blob size";
            Cd2Cw2_Op->layer->blobs()[0] = Cw2_blob;
            ops[Cd2_Cw2] = Cd2Cw2_Op;

            //************************* add all op to map
            //*******************************
            convOps[layers[i]->layer_param().name()] = ops;
        }
    }
    PERFORMANCE_STOP_RESETTING_MONITOR();

    Blob<float> *input_layer = caffe_net.input_blobs()[0];
    //    Blob<float> *output_layer = caffe_net.output_blobs()[0];
    int batch_size = input_layer->shape(0);

    LOG(WARNING) << "****Begin Warm up ******************";
    for (int i = 0; i < 5; i++) {
        inference_net_test(layers, bottom_vecs, top_vecs, convOps,
                           ABFT_CONV_OPTION);
    }
    layer_abft_op.resize(layers.size(), 1);
    if (ABFT_CONV_OPTION == 14) {
        long iteration = 0;
        layer_abft_op_time.resize(layers.size());
        for (int i = 0; i < layers.size(); i++) {
            layer_abft_op_time[i].resize(10, 0);
        }

        for (int idx = FLAGS_val_begin, idb = 0; idx < FLAGS_val_end;
             idx += FLAGS_val_stride) {
            cvReadImg(idx, input_layer, idb++);

            if (idb == batch_size) {
                ++iteration;
                LOG(WARNING) << "CONV_OP tuning, iter:" << iteration;
                layer_tuning(layers, bottom_vecs, top_vecs, convOps);
                idb = 0;

                if (iteration == 3) {
                    for (long l = 0; l < layers.size(); l++) {
                        if (layers[l]->layer_param().type() != "Convolution") {
                            continue;
                        }
                        long input_weight = bottom_vecs[l][0]->count(0);
                        long kernel_weight = layers[l]->blobs()[0]->count(0);
                        double pinput =
                                input_weight * 1.0 / (input_weight + kernel_weight);
                        double pkernel = 1 - pinput;
                        if (INJECT_ERROR == 105) {
                            pinput = 0.5;
                            pkernel = 0.5;
                        }
                        layer_abft_op[l] = 1;
                        if (layer_abft_op_time[l][3] * pkernel +
                            layer_abft_op_time[l][4] * pinput <
                            layer_abft_op_time[l][0]) {
                            layer_abft_op[l] = 3;
                        }
                        if (layer_abft_op_time[l][1] * pinput +
                            layer_abft_op_time[l][2] * pkernel <
                            layer_abft_op_time[l][0]) {
                            layer_abft_op[l] = 2;
                        }
                    }
                    break;
                }
            }
        }
    }
    LOG(WARNING) << "****End Warm up ******************";

    fault_not_detected = 0;
    fault_detected = 0;
    no_fault_false_detected = 0;
    LOG(ERROR) << "****Begin Benchmark****************";

    std::vector<double> forward_time_per_layer(layers.size(), 0.0);
    std::map<std::string, std::vector<double>> overhead_per_layer;
    std::map<std::string, double> rerun_time_per_layer;
    for (int i = 0; i < layers.size(); i++) {
        std::vector<double> overheads(5, 0.0);
        overhead_per_layer[layers[i]->layer_param().name()] = overheads;
        rerun_time_per_layer[layers[i]->layer_param().name()] = 0.0;
    }
    double forward_time = 0.0;
    double overhead_time = 0.0;
    double img_time = 0.0;
    Timer timer;
    Timer total_timer;
    total_timer.Start();
    int iteration = 0, inject_layer = 1;
    for (int idx = FLAGS_val_begin, idb = 0; idx < FLAGS_val_end;
         idx += FLAGS_val_stride) {
        timer.Start();
        cvReadImg(idx, input_layer, idb++);
        img_time += timer.MicroSeconds();

        if (idb == batch_size) {
            timer.Start();
            LOG(ERROR) << "Iteration " << ++iteration;
            while (layers[inject_layer]->layer_param().type() !=
                   "Convolution") {
                inject_layer = (inject_layer + 1) % layers.size();
            }
            inference_net(layers, bottom_vecs, top_vecs, convOps,
                          ABFT_CONV_OPTION, forward_time_per_layer,
                          overhead_per_layer, rerun_time_per_layer,
                          INJECT_ERROR ? inject_layer : -1);

            idb = 0;
            inject_layer = (inject_layer + 1) % layers.size();
            forward_time += timer.MicroSeconds();
        }
    }

    //    LOG(WARNING) << "\n\n";
    LOG(ERROR) << "***Begin Summary ***";
    LOG(ERROR) << "Average time per layer: ";
    for (long i = 0; i < layers.size(); ++i) {
        string layername = layers[i]->layer_param().name();
        double overhead_sum = 0.0;

        if (layers[i]->layer_param().type() == "Convolution" &&
            ABFT_CONV_OPTION != 0) {
            auto overheads = overhead_per_layer[layername];
            auto rerun = rerun_time_per_layer[layername];

            overhead_sum = overheads[0] + overheads[1] + overheads[2] +
                           overheads[3] + overheads[4] + rerun;
            overhead_time += overhead_sum;

            LOG(ERROR) << std::setfill(' ') << std::setw(8)
                       << std::setprecision(2) << layername << "\tForward: "
                       << (forward_time_per_layer[i] - overhead_sum) / 1000 /
                          iteration
                       << " ms."
                       << "\tOverhead: " << (overhead_sum / 1000 / iteration)
                       << " ms."
                       << "\t( Detect D: " << (overheads[0] / 1000 / iteration)
                       << " ms, "
                       << "C1: " << (overheads[1] / 1000 / iteration) << " ms, "
                       << "C2: " << (overheads[2] / 1000 / iteration) << " ms, "
                       << "C3: " << (overheads[3] / 1000 / iteration) << " ms, "
                       << "C4: " << (overheads[4] / 1000 / iteration) << " ms, "
                       << "Rerun: " << rerun / 1000 / iteration << " ms."
                       << " )";
        } else {
            LOG(ERROR) << std::setfill(' ') << std::setw(8)
                       << std::setprecision(2) << layername << "\tforward: "
                       << (forward_time_per_layer[i] - overhead_sum) / 1000 /
                          iteration
                       << " ms.";
        }
    }
    LOG(ERROR) << "Total Iterations: " << iteration;
    LOG(ERROR) << "Avg imgRead Time: " << img_time / 1000 / iteration << " ms.";
    LOG(ERROR) << "Avg Forward Time(include overhead): "
               << forward_time / 1000 / iteration << " ms.";
    LOG(ERROR) << "Ave Overhead: " << overhead_time / 1000 / iteration
               << " ms.";
    LOG(ERROR) << "Ave Overhead percentige: "
               << overhead_time / (forward_time - overhead_time) * 100 << " %";
    for (long i = 0; i < layers.size(); ++i) {
        if (layers[i]->layer_param().type() == "Convolution" &&
            ABFT_CONV_OPTION != 0) {
            string layername = layers[i]->layer_param().name();
            auto overheads = overhead_per_layer[layername];
            LOG(ERROR) << layername
                       << " Overhead Percentage:" << std::setprecision(2)
                       << overheads[0] / overhead_time << ","
                       << overheads[1] / overhead_time << ","
                       << overheads[2] / overhead_time << ","
                       << overheads[3] / overhead_time << ","
                       << overheads[4] / overhead_time << ",";
        }
    }

    vector<long> total_correction(6, 0);
    for (long i = 0; i < layers.size(); ++i) {
        if (layers[i]->layer_param().type() == "Convolution" &&
            ABFT_CONV_OPTION != 0) {
            string layername = layers[i]->layer_param().name();
            auto correction = correction_count[layername];
            for (int j = 0; j < 5; j++) {
                total_correction[j] += correction[j];
            }
            LOG(ERROR) << layername << " Correction Dist:" << correction[0]
                       << "," << correction[1] << "," << correction[2] << ","
                       << correction[3] << "," << correction[4] << ",";
        }
    }
    LOG(ERROR) << "Correction(Total) Dist:" << total_correction[0] << ","
               << total_correction[1] << "," << total_correction[2] << ","
               << total_correction[3] << "," << total_correction[4] << ",";

    for (long l = 0; l < layers.size(); l++) {
        if (layers[l]->layer_param().type() == "Convolution" &&
            ABFT_CONV_OPTION == 14) {
            long input_weight = bottom_vecs[l][0]->count(0);
            long kernel_weight = layers[l]->blobs()[0]->count(0);
            LOG(ERROR) << layers[l]->layer_param().name() << " use abft_op "
                       << std::setprecision(4) << layer_abft_op[l]
                       << ", ori time:" << layer_abft_op_time[l][0] / 1000 / 3
                       << ", input time:"
                       << layer_abft_op_time[l][1] / 1000 / 3
                       << ", input failure time:"
                       << layer_abft_op_time[l][2] / 1000 / 3
                       << ", input/kernel:" << input_weight / kernel_weight;
        }
    }
    LOG(ERROR) << "Total Time:" << total_timer.MicroSeconds() / 1000 << " ms.";
    LOG(ERROR) << "Total Image Process Time:" << img_time / 1000 << " ms.";
    LOG(ERROR) << "Total Forward Time:" << forward_time / 1000 << " ms.";

    LOG(ERROR) << "Engine: " << FLAGS_engine;
    LOG(ERROR) << "Model: " << FLAGS_model;
    LOG(ERROR) << "Threads: " << caffe_getenv("OMP_NUM_THREADS");
    LOG(ERROR) << "FT_CONV: " << ABFT_CONV_OPTION;
    LOG(ERROR) << "FT_ERROR: " << INJECT_ERROR;
    //    LOG(ERROR) << "Final SDC Rate: " << total_false_count /
    //    (total_true_count + (double) total_false_count) * 100 << " %";
    LOG(ERROR) << "fault_detected:" << fault_detected
               << ", fault_not_detected:" << fault_not_detected
               << ", false_fault:" << no_fault_false_detected;
    double precision = fault_detected /
                       (fault_detected + (double) no_fault_false_detected) * 100;
    double recall =
            fault_detected / (fault_detected + (double) fault_not_detected) * 100;
    LOG(ERROR) << "Precision: " << precision << " %";
    LOG(ERROR) << "Recall: " << recall << " %";

    LOG(ERROR) << "*** Summary END ***\n\n";

    if (profiling_rerror) {
        for (long i = 0; i < layers.size(); i++) {
            if (rerror[i][0] != 0) {
                LOG(ERROR) << "ROUND_OFF_ERROR " << i << " " << rerror_max[i][0]
                           << " " << rerror_max[i][1] << " "
                           << rerror_max[i][2];
            }
        }
    }

    //    auto batch_size = model_path[model_path.size() - 3];
    printf("Model:%s,batch_size:%d,thread:%d,fc_conv:%d,fc_error%d,overhead:%.3f,\n",
           model_name.c_str(), batch_size, caffe_getenv("OMP_NUM_THREADS"),
           ABFT_CONV_OPTION, INJECT_ERROR,
           overhead_time / (forward_time - overhead_time) * 100);
    return 0;
}

RegisterBrewFunction(inference);

int inference_baseline() {
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
    caffe::Phase phase = caffe::TEST;
    vector<string> stages = get_stages_from_flags(FLAGS_stage);

    LOG(WARNING) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);

    // Instantiate the caffe net.
    Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages, NULL,
                         FLAGS_engine);
    if (FLAGS_weights.size()) {
        caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
    }
    PERFORMANCE_INIT_MONITOR();

    const vector<shared_ptr<Layer<float>>> &layers = caffe_net.layers();
    const vector<vector<Blob<float> *>> &bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float> *>> &top_vecs = caffe_net.top_vecs();

    PERFORMANCE_STOP_RESETTING_MONITOR();

    Blob<float> *input_layer = caffe_net.input_blobs()[0];
    //    Blob<float> *output_layer = caffe_net.output_blobs()[0];
    int batch_size = input_layer->shape(0);

    Timer timer;
    timer.Start();
    double forward_time = 0.0;
    double img_time = 0.0;
    int iteration = 0;
    for (int idx = FLAGS_val_begin, idb = 0; idx < FLAGS_val_end;
         idx += FLAGS_val_stride) {
        timer.Start();
        cvReadImg(idx, input_layer, idb++);
        img_time += timer.MicroSeconds();

        if (idb == batch_size) {
            timer.Start();
            LOG(WARNING) << "Iteration " << ++iteration;
            for (int i = 0; i < layers.size(); ++i) {
                layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
            }
            idb = 0;
            forward_time += timer.MicroSeconds();
        }
    }

    LOG(ERROR) << "Avg imgRead Time: " << img_time / 1000 / iteration << " ms.";
    LOG(ERROR) << "Avg Forward Time: " << forward_time / 1000 / iteration
               << " ms.";

    return 0;
}

RegisterBrewFunction(inference_baseline);

int main(int argc, char **argv) {
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    // Set version
    gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
    // Usage message.
    gflags::SetUsageMessage(
            "command line brew\n"
            "usage: caffe <command> <args>\n\n"
            "commands:\n"
            "  inference_base     benchmark inference\n"
            "  inference     benchmark inference with abft");
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);

    if (argc == 2) {
        int ret = GetBrewFunction(caffe::string(argv[1]))();
        return ret;
    } else {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
    }
    return 0;
}
