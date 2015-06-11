#include "matrix.h"
#include "util.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
using namespace std;

vector<rnd_struct> Matrix::rnd_;
vector<Matrix> Matrix::ones_, Matrix::temp_;
vector<int> Matrix::boards_;
vector<size_t> Matrix::temp_size_, Matrix::ones_size_;
int Matrix::current_gpu_id_ = 0, Matrix::num_boards_ = 0;
map<string, long> Matrix::gpu_memory_;
Matrix Matrix::rgb_to_yuv_mat_;

Matrix::Matrix() {
  mat_.data_host = NULL;
  mat_.data_device = NULL;
  mat_.on_host = 1;
  mat_.on_device = 0;
  mat_.is_trans = 0;
  mat_.size[0] = 0;
  mat_.size[1] = 0;
  mat_.owns_data = 0;
  mat_.tex_obj = 0;
  SetupTranspose();
  shape_.shape[0] = 0;
  shape_.shape[1] = 0;
  shape_.shape[2] = 0;
  shape_.shape[3] = 0;
}

Matrix::~Matrix() {
  destroy_tex(&mat_);  // Does a check for texture creation.
  if (mat_.owns_data == 1) {
    if(mat_.data_host != NULL) free(mat_.data_host);
    free_device_memory(&mat_);
    auto it = gpu_memory_.find(name_);
    if (it != gpu_memory_.end()) {
      it->second = 0;
    }
  }
}

Matrix::Matrix(const size_t rows, const size_t cols, const bool on_gpu) {
  Matrix();
  if (on_gpu) {
    AllocateGPUMemory(rows, cols);
  } else {
    AllocateMainMemory(rows, cols);
  }
}

void Matrix::SetShape4D(int d1, int d2, int d3, int d4) {
  shape_.shape[0] = d1;
  shape_.shape[1] = d2;
  shape_.shape[2] = d3;
  shape_.shape[3] = d4;
}

void Matrix::SetShape4D_like(Matrix& mat) {
 Shape4D s = mat.GetShape4D();
 SetShape4D(s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
}

Shape4D& Matrix::GetShape4D() {
  return shape_;
}

void Matrix::GetSlice(Matrix& slice, size_t start, size_t end) {
  get_slice(&mat_, slice.GetMat(), start, end);
  slice.SetupTranspose();
  slice.SetGPUId(gpu_id_);
}

void Matrix::Tie(Matrix &m) {
  cout << "Tying" << endl;
  mat_ = *(m.GetMat());
  mat_t_ = *(m.GetMatTranspose());
}

void Matrix::SetupTranspose() {
  mat_t_ = mat_;
  mat_t_.is_trans = 1 - mat_.is_trans;
}

void Matrix::AllocateGPUMemory(const size_t rows, const size_t cols) {
  AllocateGPUMemory(rows, cols, "");
}

void Matrix::AllocateGPUMemory(const size_t rows, const size_t cols, const string& name) {
  if (rows != mat_.size[0] || cols != mat_.size[1]) {
    name_ = name;
    gpu_id_ = current_gpu_id_;
    if (gpu_id_ < 0 || gpu_id_ >= num_boards_) {
      cerr << "This should not happen" << endl;
      exit(1);
    }
    if (GetNumEls() > 0) {
      Matrix::gpu_memory_[name] -= GetNumEls() * sizeof(float);
      free_device_memory(&mat_);
    }
    AllocateMainMemory(rows, cols);
    CopyToDevice();
    mat_.owns_data = 1;
    SetupTranspose();
    //const size_t size = (rows * cols * sizeof(float)) >> 20;
    //cout << "Allocated GPU memory " << rows << " * " << cols << " " << size << "MB for " << name << endl;
    cuda_create_event(&ready_);
    Matrix::gpu_memory_[name] += GetNumEls() * sizeof(float);
  }
}

void Matrix::ShowMemoryUsage() {
  map<string, long> summary;
  map<string, string> groups;
  groups["deriv"] = "layer";
  groups["state"] = "layer";
  groups["data"] = "layer";
  groups["bias"] = "edge";
  groups["optimizer"] = "edge";
  groups["weight"] = "edge";
  groups[""] = "misc";
  for(auto it : Matrix::gpu_memory_) {
    const string& name = it.first;
    size_t pos = name.find_last_of(" _");
    string group = (pos != string::npos) ? name.substr(pos+1) : name;
    //cout << name << " Group " << group << " " << it.second / (1024.0 * 1024) << " MB" << endl;
    auto lookup = groups.find(group);
    string key = (lookup == groups.end()) ? group : lookup->second;
    summary[key] += it.second;
  }
  cout << "----- SUMMARY ------" << endl;
  long total = 0;
  for(auto it : summary) {
    cout << it.first << "\t" << it.second / (1024.0 * 1024) << " MB" << endl;
    total += it.second;
  }
  cout << "TOTAL " << "\t" << total / (1024.0 * 1024) << " MB" << endl;
}

void Matrix::AllocateMainMemory(const size_t rows, const size_t cols) {
  if (mat_.data_host != NULL) free(mat_.data_host);
  mat_.data_host = (float*)calloc(rows * cols, sizeof(float));
  if (mat_.data_host == NULL) {
    cerr << "Error: Could not allocate main memory for matrix of size "
         << rows << " by " << cols << "." << endl;
    exit(1);
  }
  mat_.size[0] = rows;
  mat_.size[1] = cols;
  mat_.on_device = 0;
  mat_.on_host = 1;
  mat_.is_trans = 0;
  mat_.owns_data = 1;
}

void Matrix::CopyToDevice() {
  CopyToDeviceSlice(0, mat_.size[1]);
}

void Matrix::CopyToHost() {
  CopyToHostSlice(0, mat_.size[1]);
}

void Matrix::CopyFromMainMemory(Matrix& mat) {
  float* src = mat.GetHostData();
  float* dest = GetHostData();
  memcpy(dest, src, sizeof(float) * GetNumEls());
}

void Matrix::Set(const float val) {
  int err_code = assign_scalar(&mat_, val);
  if (err_code != 0) {
    cerr << "Error: Could not set to scalar : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::Set(Matrix& val) {
  int err_code = copy_on_device(val.GetMat(), &mat_);  // source, dest.
  if (err_code != 0) {
    cerr << "Error: Could not set to val : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::WriteValue(int index, float val) {
  WriteValue(index % mat_.size[0], index / mat_.size[0], val);
}

float Matrix::ReadValue(int index) {
  return ReadValue(index % mat_.size[0], index / mat_.size[0]);
}

void Matrix::WriteValue(int row, int col, float val) {
  int err_code = write_at(&mat_, row, col, val);
  if (err_code != 0) {
    cerr << "Error: Could not write value : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

float Matrix::ReadValue(int row, int col) {
  int err_code;
  float res = read_from(&mat_, row, col, &err_code);
  if (err_code != 0) {
    cerr << "Error: Could not read value : " << GetStringError(err_code) << endl;
    exit(1);
  }
  return res;
}

void Matrix::CopyP2PAsync(Matrix& val) {
  int err_code = copy_on_device_p2p_async(val.GetMat(), &mat_, val.GetGPUId(), gpu_id_);  // source, dest.
  if (err_code != 0) {
    cerr << "Error: Could not copy async : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::FillWithRandn() {
  if (gpu_id_ != current_gpu_id_) {
    cerr << "GPU mismatch " << endl;
    exit(1);
  }
  int err_code = fill_with_randn(&rnd_[current_gpu_id_], &mat_);
  if (err_code != 0) {
    cerr << "Error: Could not fill with randn : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::FillWithRand() {
  int err_code = fill_with_rand(&rnd_[current_gpu_id_], &mat_);
  if (err_code != 0) {
    cerr << "Error: Could not fill with rand : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::SampleBernoulli(float val) {
  int err_code = assign_scalar(&mat_, val);
  err_code = sample_bernoulli(&rnd_[current_gpu_id_], &mat_, &mat_);
  if (err_code != 0) {
    cerr << "Error: Could not fill with rand : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

float Matrix::Sum() {
  Matrix ones;
  size_t rows = mat_.size[0];
  size_t cols = mat_.size[1];
  reshape(&mat_, 1, -1);
  GetOnes(1, rows * cols, ones);
  int err = 0;
  float res = vdot(ones.GetMat(), &mat_, &err);
  if (err != 0) {
    cerr << "Error in vdot " << GetStringError(err) << endl;
    cerr << "Summing matrix of shape " << rows << " " << cols << endl;
    exit(1);
  }
  reshape(&mat_, rows, cols);
  return res;
}

void Matrix::Add(Matrix& m) {
  add_elementwise(&mat_, m.GetMat(), &mat_);
}

// self += alpha * m
void Matrix::Add(Matrix& m, float alpha) {
  add_mult(&mat_, m.GetMat(), alpha);
}

void Matrix::CopyTransposeBig(Matrix& m) {
  copy_transpose_big_matrix(&mat_, m.GetMat());
}

void Matrix::CopyTranspose(Matrix& m) {
  copy_transpose(&mat_, m.GetMat());
}

float Matrix::VDot(Matrix& m) {
  int err;
  return vdot(&mat_, m.GetMat(), &err);
}

float Matrix::EuclidNorm() {
  int err_code;
  float res = euclid_norm(&mat_, &err_code);
  return res;
}

void Matrix::Subtract(Matrix& m, Matrix& target) {
  int err_code = subtract_elementwise(&mat_, m.GetMat(), target.GetMat());
  if (err_code != 0) {
    cerr << "Error in subtract." << endl;
    exit(1);
  }
}

void Matrix::CopyToDeviceSlice(const size_t start, const size_t end) {
  int err_code = copy_to_device_slice(&mat_, start, end);
  if (err_code != 0) {
    cerr << "Error copying matrix of size " << mat_.size[0] << " "
         << mat_.size[1] << " slice " << start << ":" << end << " to device: "
         << GetStringError(err_code) << endl;
    exit(1);
  } else {
    //cout << "Successfully copied matrix of size " << mat_.size[0] << " " << mat_.size[1] << " to device." << endl;
  }
}

void Matrix::CopyToHostSlice(const size_t start, const size_t end) {
  int err_code = copy_to_host_slice(&mat_, start, end);
  if (err_code != 0) {
    cerr << "Error copying matrix of size " << mat_.size[0] << " "
         << mat_.size[1] << " slice " << start << ":" << end << " to host: "
         << GetStringError(err_code) << endl;
    exit(1);
  } else {
    //cout << "Successfully copied matrix of size " << mat->size[0] << " " << mat->size[1] << " to device." << endl;
  }
}


void Matrix::Reshape(size_t rows, size_t cols) {
  reshape(&mat_, rows, cols);
  mat_t_ = mat_;
  mat_t_.is_trans = 1;
}

void Matrix::PrintToFile(const string& filename) {
  int err_code = copy_to_host(&mat_);
  if (err_code != 0) {
    cerr << "Error: Could not copy to host : " << GetStringError(err_code) << endl;
    exit(1);
  }
  ofstream f(filename.c_str(), ios::out);
  for (int i = 0; i < mat_.size[0]; i++) {
    for (int j = 0; j < mat_.size[1]; j++) {
      f << mat_.data_host[j * mat_.size[0] + i] << " ";
    }
    f << '\n';
  }
  f.close();
}

void Matrix::Print() {
  cout << "Printing matrix on gpu " << gpu_id_ << " current gpu " << current_gpu_id_ << endl;
  int err_code = copy_to_host(&mat_);
  if (err_code != 0) {
    cerr << "Error: Could not copy to host : " << GetStringError(err_code) << endl;
    exit(1);
  }
  cout << setprecision(12);
  for (size_t i = 0; i < mat_.size[0]; i++) {
    if (i < 10) {
      for (size_t j = 0; j < mat_.size[1]; j++) {
        if (j < 10) {
          cout << mat_.data_host[j * mat_.size[0] + i] << " " ;
        } else {
          cout << ". . . ";
          break;
        }
      }
    } else {
      cout << ". . .\n";
      break;
    }
    cout << '\n';
  }
}

bool Matrix::CheckNaN() {
  CopyToHost();
  float* data = mat_.data_host;
  bool is_nan = false;
  size_t i = 0;
  for (; i < mat_.size[0] * mat_.size[1] && !is_nan; i++) {
    is_nan = !isfinite(data[i]);
  }
  if (is_nan) cout << "Nan at location " << i << " row " << i % mat_.size[0] << " col " << i / mat_.size[0] << endl;
  return is_nan;
}

string Matrix::GetShapeString() {
  stringstream ss;
  ss << mat_.size[0] << " " << mat_.size[1];
  return ss.str();
}

string Matrix::GetShape4DString() {
  stringstream ss;
  ss << shape_.shape[0] << " " << shape_.shape[1] << " " << shape_.shape[2] << " " << shape_.shape[3];
  return ss.str();
}

void Matrix::WriteToFile(FILE* file) {
  copy_to_host(&mat_);
  fwrite(mat_.size, sizeof(int), 2, file);
  fwrite(mat_.data_host, sizeof(float), mat_.size[0] * mat_.size[1], file);
}

void Matrix::ReadFromFile(FILE* file) {
  fread(mat_.size, sizeof(int), 2, file);
  fread(mat_.data_host, sizeof(float), mat_.size[0] * mat_.size[1], file);
  int err_code = copy_to_device(&mat_);
  if (err_code != 0) {
    cerr << "Error copying matrix to device : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::WriteHDF5(hid_t file, const string& name) {
  copy_to_host(&mat_);
  // cols, rows swapped because cudamat is col major, but hdf5 is row major.
  WriteHDF5CPU(file, mat_.data_host, mat_.size[1], mat_.size[0], name);
}

void Matrix::ReadHDF5(hid_t file, const string& name) {
  ReadHDF5CPU(file, mat_.data_host, mat_.size[0] * mat_.size[1], name);
  copy_to_device(&mat_);
}

void Matrix::AllocateAndReadHDF5(hid_t file, const string& name) {
  int rows, cols;
  ReadHDF5Shape(file, name, &rows, &cols);
  AllocateGPUMemory(rows, cols);
  ReadHDF5(file, name);
}

void Matrix::GetOnes(size_t rows, size_t cols, Matrix& ones) {
  Matrix& o = Matrix::ones_[current_gpu_id_];
  size_t size = o.GetCols();
  if (size == 0) {  // Allocate memory on first call to GetOnes.
    o.AllocateGPUMemory(1, ones_size_[current_gpu_id_], "ones");
    o.Set(1);
    size = ones_size_[current_gpu_id_];
  }
  if (rows * cols > size) {
    cerr << "Ones has only " << size << " elements. Requested was "
         << rows << " * " << cols << endl;
    exit(1);
  }
  o.GetSlice(ones, 0, rows * cols);
  ones.Reshape(rows, cols);
}

void Matrix::GetTemp(size_t rows, size_t cols, Matrix& temp) {
  Matrix& t = Matrix::temp_[current_gpu_id_];
  size_t size = t.GetNumEls();
  const size_t length = rows * cols;
  if (length > size) {  // Allocate memory as required.
    size_t temp_size = temp_size_[current_gpu_id_];
    if (length > temp_size) {
      temp_size = length;
      temp_size_[current_gpu_id_] = length;
    }
    cout << "Allocating new temp memory of size " << temp_size << " on gpu " << current_gpu_id_ << endl;
    t.AllocateGPUMemory(1, temp_size, "temp");
    //cout << "Allocated " << (temp_size_[current_gpu_id_] >> 18) << " MB for temp." << endl;
  }
  t.GetSlice(temp, 0, length);
  reshape(temp.GetMat(), rows, cols);
}

float Matrix::Norm() {
  int err_code;
  float res = euclid_norm(&mat_, &err_code);
  if (err_code != 0) {
    cerr << "Error in Matrix::Norm " << GetStringError(err_code) << endl;
    exit(1);
  }
  return res;
}

void Matrix::SquashRelu() {
  apply_relu_squash(&mat_, &mat_, 2);
}

void Matrix::SetupCUDADevices(const vector<int>& boards) {
  int err_code;
  num_boards_ = boards.size();
  bool check_p2p_fermi  = num_boards_ > 1;
  for (int i = 0; i < num_boards_; i++) {
    boards_.push_back(boards[i]);
    err_code = cuda_set_device(boards[i]);
    if (err_code != 0) {
      cerr << "Error setting device id! " << GetStringError(err_code) << endl;
      exit(1);
    }
    if (check_p2p_fermi && !cuda_is_fermi(boards[i])) {
      cerr << "Error : Board is not Fermi! " << GetStringError(err_code) << endl;
      exit(1);
    }
  }
  if (check_p2p_fermi) {
    // Setup P2P.
    err_code = cuda_set_P2P(boards[0], boards[1]);
    if (err_code != 0) {
      cerr << "Warning : Could not set up P2P, GPU-to-GPU communication will be slow. "
           << GetStringError(err_code) << endl;
    }
  }
  err_code = cublas_init();
  if (err_code != 0) {
    cerr << "Error initializing cublas!" << GetStringError(err_code) << endl;
    exit(1);
  }
  temp_size_.resize(num_boards_);
  ones_size_.resize(num_boards_);
  rnd_.resize(num_boards_);
  ones_.resize(num_boards_);
  temp_.resize(num_boards_);

  for (int i = 0; i < num_boards_; i++) {
    temp_size_[i] = 10000;
    ones_size_[i] = 1024 * 512;
  }
  SetDevice(0);
}

void Matrix::SetupCUDADevice(int board) {
  vector<int> boards;
  boards.push_back(board);
  SetupCUDADevices(boards);
}

int Matrix::GetDevice() {
  int board;
  cudaError_t err = cudaGetDevice(&board);
  if (err != cudaSuccess) {
    cerr << "Could not get which board is current." << endl;
    exit(1);
  }
  for (int i = 0; i < num_boards_; i++) {
    if (boards_[i] == board) return i;
  }
  cerr << "current board was not set" << endl;
  exit(1);
  return 0;
}

void Matrix::SetDevice(int gpu_id) {
  if (num_boards_ < 2) return;
  if (current_gpu_id_ == gpu_id) return;
  int err_code = cuda_set_device(boards_[gpu_id]);
  if (err_code != 0) {
    cerr << "Error setting device id! " << GetStringError(err_code) << endl;
    exit(1);
  }
  current_gpu_id_ = gpu_id;
}
/*
void Matrix::SyncDevice(int gpu_id) {
  if (num_boards_ < 2) return;
  int old_id = current_gpu_id_;
  SetDevice(gpu_id);
  cuda_sync_threads();
  SetDevice(old_id);
}
*/

void Matrix::SyncAllDevices() {
  int current_gpu_backup = current_gpu_id_;
  for (int i = 0; i < num_boards_; i++) {
    SetDevice(i);
    cuda_sync_threads();
  }
  SetDevice(current_gpu_backup);
}

void Matrix::InitRandom(int seed){
  int err_code;
  for (int i = 0; i < num_boards_; i++) {
    SetDevice(i);
    err_code = init_random(&rnd_[i], seed + i);
    if (err_code != 0) {
      cerr << "Error init random board " << i << " " << GetStringError(err_code) << endl;
      exit(1);
    }
  }
}

void Matrix::RegisterTempMemory(size_t size, const string& why) {
  if (size > temp_size_[current_gpu_id_]) {
    temp_size_[current_gpu_id_] = size;
    cout << "Max for " << why << " " << size << " on gpu " << current_gpu_id_ << endl;
  }
}

void Matrix::RegisterTempMemory(size_t size) {
  RegisterTempMemory(size, "");
}

void Matrix::RegisterOnes(size_t size) {
  if (size > ones_size_[current_gpu_id_]) {
    ones_size_[current_gpu_id_] = size;
  }
}

void Matrix::SetReady() {
  if (num_boards_ < 2) return;

  int err_code;
  if (current_gpu_id_ != gpu_id_) {
    cerr << "Error: Current gpu " << current_gpu_id_ << " must be same as the"
         << " one on which the event was created" << gpu_id_ << "." << endl;
    exit(1);
  }
  err_code = cuda_record_event(&ready_);
  if (err_code != 0) {
    cerr << "Error: Could not set ready." << endl;
    exit(1);
  }
}

void Matrix::WaitTillReady() {
  if (num_boards_ < 2) return;

  int err_code = cuda_synchronize_event(&ready_);
  if (err_code != 0) {
    cerr << "Error: Could not synchronize." << endl;
    exit(1);
  }
}


void Matrix::AddRowVec(Matrix& v) {
  add_row_vec(&mat_, v.GetMat(), &mat_);
}

void Matrix::AddRowVec(Matrix& v, float alpha) {
  add_row_mult(&mat_, v.GetMat(), &mat_, alpha);
}

void Matrix::AddColVec(Matrix& v, float alpha) {
  add_col_mult(&mat_, v.GetMat(), &mat_, alpha);
}

void Matrix::DivideByColVec(Matrix& v) {
  div_by_col_vec(&mat_, v.GetMat(), &mat_);
}

void Matrix::DivideByRowVec(Matrix& v) {
  div_by_row_vec(&mat_, v.GetMat(), &mat_);
}

// self *= val
void Matrix::Mult(float val) {
  mult_by_scalar(&mat_, val, &mat_, 0);
}
void Matrix::Mult(Matrix& val) {
  mult_elementwise(&mat_, val.GetMat(), &mat_, 0);
}
void Matrix::MultByRowVec(Matrix& val) {
  mult_by_row_vec(&mat_, val.GetMat(), &mat_);
}
void Matrix::Divide(float val) {
  divide_by_scalar(&mat_, val, &mat_);
}
void Matrix::Divide(Matrix& val) {
  divide_elementwise(&mat_, val.GetMat(), &mat_);
}

void Matrix::Add(float val) {
  add_scalar(&mat_, val, &mat_);
}

void Matrix::Sqrt() {
  apply_sqrt(&mat_, &mat_);
}
// c = alpha * c + beta * a * b
void Matrix::Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta) {
  dot(a.GetMat(), b.GetMat(), c.GetMat(), alpha, beta);
}

// c = alpha * c + beta * T(a) * T(b)
void Matrix::Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta,
                 bool transpose_a, bool transpose_b) {
  cudamat* a_mat = transpose_a ? a.GetMatTranspose() : a.GetMat();
  cudamat* b_mat = transpose_b ? b.GetMatTranspose() : b.GetMat();
  dot(a_mat, b_mat, c.GetMat(), alpha, beta);
}

void Matrix::Dropout(float dropprob, float fill_value, float scale_factor) {
  dropout(&Matrix::rnd_[gpu_id_], &mat_, dropprob, fill_value, scale_factor);
}

void Matrix::UpperBoundMod(float val) {
  upper_bound_mod_scalar(&mat_, val, &mat_);
}

void Matrix::LowerBound(float val) {
  lower_bound_scalar(&mat_, val, &mat_);
}

void Matrix::ApplyDerivativeOfReLU(Matrix& state) {
  apply_rectified_linear_deriv(&mat_, state.GetMat(), &mat_);
}

void Matrix::ApplySoftmax() {
  softmax_row_major_multi(&mat_, GetCols());
}

void Matrix::ApplyLogistic() {
  apply_sigmoid(&mat_, &mat_);
}

void Matrix::ApplyDerivativeOfLogistic(Matrix& state) {
  apply_logistic_deriv(&mat_, state.GetMat(), &mat_);
}

void Matrix::LogisticCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv) {
  apply_logistic_grad(state.GetMat(), gt.GetMat(), deriv.GetMat());
}

void Matrix::LogisticCorrect(Matrix& state, Matrix& gt, Matrix& output) {
  get_logistic_correct_normalized(state.GetMat(), gt.GetMat(), output.GetMat());
}

void Matrix::SoftmaxCorrect(Matrix& state, Matrix& gt, Matrix& output) {
  get_softmax_correct_row_major(state.GetMat(), gt.GetMat(), output.GetMat());
}

void Matrix::SoftmaxCE(Matrix& state, Matrix& gt, Matrix& output) {
  get_softmax_cross_entropy_row_major(state.GetMat(), gt.GetMat(), output.GetMat(), 1e-10);
}

void Matrix::SoftmaxCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv) {
  apply_softmax_grad_row_major(state.GetMat(), gt.GetMat(), deriv.GetMat());
}

void Matrix::SoftmaxDistCE(Matrix& state, Matrix& gt, Matrix& output) {
  int err = compute_cross_entropy(gt.GetMat(), state.GetMat(), output.GetMat(), 1e-10);
  if (err != 0) {
    cerr << "SoftmaxDistCE Error : " << GetStringError(err) << endl;
    exit(1);
  }
}

void Matrix::HingeLossDeriv(Matrix& state, Matrix& gt, Matrix& deriv, bool quadratic, float margin) {
  int err_code = hinge_loss_row_major(state.GetMat(), gt.GetMat(), deriv.GetMat(), quadratic, margin);
  if (err_code != 0) {
    cerr << "Error in hinge loss " << GetStringError(err_code) << endl;
    exit(1);
  }
}

// target = alpha * target + beta * sum_rows(self)
void Matrix::SumRows(Matrix& target, float alpha, float beta) {
  sum_by_axis(&mat_, target.GetMat(), 0, beta, alpha);
  //Matrix ones;
  //Matrix::GetOnes(1, GetRows(), ones);
  //dot(ones.GetMat(), &mat_, target.GetMat(), alpha, beta);
}

// target = alpha * target + beta * sum_cols(self)
void Matrix::SumCols(Matrix& target, float alpha, float beta) {
  sum_by_axis(&mat_, target.GetMat(), 1, beta, alpha);  // Optimized for large number of rows.
  //Matrix ones;
  //Matrix::GetOnes(GetCols(), 1, ones);
  //dot(&mat_, ones.GetMat(), target.GetMat(), alpha, beta);
}

// target = alpha * target + beta * sum_cols(self**2)
void Matrix::SqSumAxis(Matrix& target, int axis, float beta, float alpha) {
  int err_code = sqsum_by_axis(&mat_, target.GetMat(), axis, beta, alpha);
  if (err_code != 0) {
    cerr << "Error in sqsum_by_axis " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::NormLimitByAxis(int axis, float val, bool constraint) {
  normlimit_by_axis(&mat_, &mat_, axis, val, constraint);
}

void Matrix::NormalizeColumnwise() {
  normalize_by_axis(&mat_, &mat_, 0);
}

void Matrix::ShuffleColumns(Matrix& rand_perm_indices) {
  shuffleColumns(&mat_, rand_perm_indices.GetMat());
}

void Matrix::ConvUp(Matrix& input, Matrix& w, Matrix& output,
                    ConvDesc conv_desc, float scale_targets) {
#ifdef USE_GEMM
  convUpGemm(
#else
  convUp(
#endif
      input.GetMat(), w.GetMat(), output.GetMat(), &input.GetShape4D(),
         &w.GetShape4D(), &output.GetShape4D(), conv_desc, scale_targets);
}

void Matrix::Conv3DUp(Matrix& input, Matrix& w, Matrix& output,
                      ConvDesc conv_desc, float scale_targets) {
#ifdef USE_GEMM
  convUp3DGemm(input.GetMat(), w.GetMat(), output.GetMat(), &input.GetShape4D(),
               &w.GetShape4D(), &output.GetShape4D(), conv_desc, scale_targets);
#else
  cerr << "GEMM kernels required for 3D convolutions." << endl;
  exit(1);
#endif
}


void Matrix::ConvDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                      ConvDesc conv_desc, float scale_targets) {
#ifdef USE_GEMM
  convDownGemm(
#else
  convDown(
#endif
    deriv_output.GetMat(), w.GetMat(), deriv_input.GetMat(),
           &deriv_output.GetShape4D(), &w.GetShape4D(), &deriv_input.GetShape4D(),
           conv_desc, scale_targets);
}

void Matrix::Conv3DDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                        ConvDesc conv_desc, float scale_targets) {
#ifdef USE_GEMM
  convDown3DGemm(deriv_output.GetMat(), w.GetMat(), deriv_input.GetMat(),
           &deriv_output.GetShape4D(), &w.GetShape4D(), &deriv_input.GetShape4D(),
           conv_desc, scale_targets);
#else
  cerr << "GEMM kernels required for 3D convolutions." << endl;
  exit(1);
#endif
}

void Matrix::ConvOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                      ConvDesc conv_desc, int partial_sum_y, int partial_sum_x,
                      float scale_targets, float scale_outputs) {
#ifdef USE_GEMM
  convOutpGemm(input.GetMat(), deriv_output.GetMat(), dw.GetMat(),
           &input.GetShape4D(), &deriv_output.GetShape4D(), &dw.GetShape4D(),
           conv_desc, scale_targets, scale_outputs);
#else
  convOutp(input.GetMat(), deriv_output.GetMat(), dw.GetMat(),
           &input.GetShape4D(), &deriv_output.GetShape4D(), &dw.GetShape4D(),
           conv_desc, partial_sum_y, partial_sum_x, scale_targets,
           scale_outputs);
#endif
}

void Matrix::Conv3DOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                        ConvDesc conv_desc, float scale_targets, float scale_outputs) {
#ifdef USE_GEMM
  convOutp3DGemm(input.GetMat(), deriv_output.GetMat(), dw.GetMat(),
           &input.GetShape4D(), &deriv_output.GetShape4D(), &dw.GetShape4D(),
           conv_desc, scale_targets, scale_outputs);
#else
  cerr << "GEMM kernels required for 3D convolutions." << endl;
  exit(1);
#endif
}

void Matrix::LocalUp(Matrix& input, Matrix& w, Matrix& output,
                    ConvDesc conv_desc, float scale_targets) {
#ifdef USE_GEMM
  localUpGemm(
#else
  localUp(
#endif
      input.GetMat(), w.GetMat(), output.GetMat(), &input.GetShape4D(),
         &w.GetShape4D(), &output.GetShape4D(), conv_desc, scale_targets);
}

void Matrix::LocalDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                     ConvDesc conv_desc, float scale_targets) {
#ifdef USE_GEMM
  localDownGemm(
#else
  localDown(
#endif
   deriv_output.GetMat(), w.GetMat(), deriv_input.GetMat(),
   &deriv_output.GetShape4D(), &w.GetShape4D(), &deriv_input.GetShape4D(),
   conv_desc, scale_targets);
}

void Matrix::LocalOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                      ConvDesc conv_desc,
                      float scale_targets, float scale_outputs) {
#ifdef USE_GEMM
  localOutpGemm(
#else
  localOutp(
#endif
    input.GetMat(), deriv_output.GetMat(), dw.GetMat(),
    &input.GetShape4D(), &deriv_output.GetShape4D(), &dw.GetShape4D(),
    conv_desc, scale_targets, scale_outputs);
}

void Matrix::ConvMaxPool(Matrix& input, Matrix& output, ConvDesc conv_desc) {
#ifdef USE_GEMM
  MaxPoolGemm(input.GetMat(), output.GetMat(), &input.GetShape4D(),
          &output.GetShape4D(), conv_desc, 0, 1);
#else
  MaxPool(input.GetMat(), output.GetMat(), &input.GetShape4D(),
          &output.GetShape4D(), conv_desc);
#endif
}

void Matrix::ConvAvgPool(Matrix& input, Matrix& output, ConvDesc conv_desc) {
#ifdef USE_GEMM
  AvgPoolGemm(input.GetMat(), output.GetMat(), &input.GetShape4D(),
              &output.GetShape4D(), conv_desc, 0, 1);
#else
  AvgPool(input.GetMat(), output.GetMat(), &input.GetShape4D(),
          &output.GetShape4D(), conv_desc);
#endif
}

void Matrix::ConvMaxPoolUndo(Matrix& input, Matrix& deriv_output, Matrix& output,
                             Matrix& deriv_input, ConvDesc conv_desc,
                             float scale_targets) {
#ifdef USE_GEMM
  MaxPoolUndoGemm(
#else
  MaxPoolUndo(
#endif
      input.GetMat(), deriv_output.GetMat(), output.GetMat(),
              deriv_input.GetMat(), &input.GetShape4D(),
              &deriv_output.GetShape4D(), conv_desc, scale_targets);
}

void Matrix::ConvAvgPoolUndo(Matrix& input, Matrix& deriv_output,
                             ConvDesc conv_desc, float scale_targets) {
#ifdef USE_GEMM
  AvgPoolUndoGemm(
#else
  AvgPoolUndo(
#endif
    input.GetMat(), deriv_output.GetMat(), &input.GetShape4D(),
              &deriv_output.GetShape4D(), conv_desc, scale_targets);
}

void Matrix::ConvResponseNormCrossMap(
    Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
    float powScale, bool blocked) {
#ifdef USE_GEMM
  ResponseNormCrossMapGemm(
#else
  ResponseNormCrossMap(
#endif
    input.GetMat(), output.GetMat(), numFilters, sizeF, addScale, powScale,
    blocked);
}

void Matrix::ConvResponseNormCrossMap3D(
    Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
    float powScale, bool blocked, int image_size_t) {
#ifdef USE_GEMM
  ResponseNormCrossMap3DGemm(input.GetMat(), output.GetMat(), numFilters, sizeF,
                             addScale, powScale, blocked, image_size_t);
#else
  cerr << "GEMM kernels required for 3D convolutions." << endl;
  exit(1);
#endif
}

void Matrix::ConvResponseNormCrossMapUndo(
    Matrix& outGrads, Matrix& inputs, Matrix& acts, Matrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked) {

#ifdef USE_GEMM
  ResponseNormCrossMapUndoGemm(outGrads.GetMat(), inputs.GetMat(), targets.GetMat(),
                           numFilters, sizeF, addScale, powScale, blocked);
#else
  ResponseNormCrossMapUndo(outGrads.GetMat(), inputs.GetMat(), acts.GetMat(),
                           targets.GetMat(), numFilters, sizeF, addScale, powScale,
                           blocked);
#endif
}

void Matrix::ConvResponseNormCrossMapUndo3D(
    Matrix& outGrads, Matrix& inputs, Matrix& acts, Matrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked, int image_size_t) {

#ifdef USE_GEMM
  ResponseNormCrossMap3DUndoGemm(
      outGrads.GetMat(), inputs.GetMat(), targets.GetMat(),
      numFilters, sizeF, addScale, powScale, blocked, image_size_t);
#else
  cerr << "GEMM kernels required for 3D convolutions." << endl;
  exit(1);
#endif
}


void Matrix::ConvUpSample(Matrix& input, Matrix& output, int factor,
                          float scaleTargets) {
#ifdef USE_GEMM
  UpSampleGemm(
#else
  UpSample(
#endif
      input.GetMat(), output.GetMat(), &input.GetShape4D(),
           &output.GetShape4D(), factor, scaleTargets);
}

void Matrix::ConvDownSample(Matrix& input, Matrix& output, int factor) {
#ifdef USE_GEMM
  DownSampleGemm(
#else
  DownSample(
#endif
    input.GetMat(), output.GetMat(), &input.GetShape4D(),
    &output.GetShape4D(), factor);
}

void Matrix::ConvRGBToYUV(Matrix& input, Matrix& output) {
  if (Matrix::rgb_to_yuv_mat_.GetNumEls() == 0) {
    Matrix::rgb_to_yuv_mat_.AllocateGPUMemory(3, 3);
    float* data = Matrix::rgb_to_yuv_mat_.GetHostData();
    data[0] = 0.2126f;    data[1] = 0.7152f;    data[2] = 0.0722f;
    data[3] = -0.09991f;  data[4] = -0.33609f;  data[5] = 0.436f;
    data[6] = 0.615f;     data[7] = -0.55861f;  data[8] = -0.05639f;
    Matrix::rgb_to_yuv_mat_.CopyToDevice();
  }
  int batch_size = input.GetRows();
  input.Reshape(-1, 3);
  output.Reshape(-1, 3);
  dot(input.GetMat(), Matrix::rgb_to_yuv_mat_.GetMat(), output.GetMat(), 1, 0);
  output.Reshape(batch_size, -1);
  input.Reshape(batch_size, -1);
}

void Matrix::ExtractPatches(Matrix& source, Matrix& dest, Matrix& width_offset,
                            Matrix& height_offset, Matrix& flip_bit,
                            int image_size_y, int image_size_x, int patch_size_y,
                            int patch_size_x) {
  int err_code = extract_patches(source.GetMat(), dest.GetMat(),
                                 width_offset.GetMat(), height_offset.GetMat(),
                                 flip_bit.GetMat(), image_size_x, image_size_y,
                                 patch_size_x, patch_size_y);
  if (err_code != 0) {
    cerr << "Error extracting patches " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::AddToEachPixel(Matrix& v, float mult) {
  add_to_each_pixel(&mat_, v.GetMat(), &mat_, mult);
}

void Matrix::RectifyBBox(Matrix& width_offset, Matrix& height_offset,
                         Matrix& flip, int patch_width, int patch_height) {

  int err_code = rectify_bounding_boxes(
      &mat_, width_offset.GetMat(), height_offset.GetMat(), flip.GetMat(),
      patch_width, patch_height);

  if (err_code != 0) {
    cerr << "Error rectifying boxes " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::AdagradUpdate(Matrix& adagrad_history, Matrix& gradient, float delta) {
  int err_code = adagrad(adagrad_history.GetMat(), gradient.GetMat(), delta);
  if (err_code != 0) {
    cerr << "Error in adagrad update " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::RMSPropUpdate(Matrix& rms_history, Matrix& gradient, float factor) {
  int err_code = rms_prop(rms_history.GetMat(), gradient.GetMat(), factor);
  if (err_code != 0) {
    cerr << "Error in rms update " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::BNBprop(Matrix& deriv, Matrix& input, Matrix& gamma, Matrix& mu,
                     Matrix& sigma, Matrix& target, float scale_targets) {
  int err_code = bn_bprop(deriv.GetMat(), input.GetMat(), gamma.GetMat(),
                          mu.GetMat(), sigma.GetMat(), target.GetMat(), scale_targets);
  if (err_code != 0) {
    cerr << "Error in BNBprop " << GetStringError(err_code) << endl;
    exit(1);
  }
}


void Matrix::BNGrad(Matrix& deriv, Matrix& input, Matrix& mu, Matrix& sigma,
                    Matrix& dgamma, Matrix& dbeta) {
  int err_code = bn_grad(deriv.GetMat(), input.GetMat(), mu.GetMat(),
                         sigma.GetMat(), dgamma.GetMat(), dbeta.GetMat());
  if (err_code != 0) {
    cerr << "Error in BNGrad " << GetStringError(err_code) << endl;
    exit(1);
  }
}
