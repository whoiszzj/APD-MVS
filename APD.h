#ifndef _APD_H_
#define _APD_H_
#include "main.h"

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)
#define M_PI 3.14159265358979323846

using namespace boost::filesystem;

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);

void CudaCheckError(const char* file, const int line);

bool ReadBinMat(const path &mat_path, cv::Mat &mat);

bool WriteBinMat(const path &mat_path, const cv::Mat &mat);

bool ReadCamera(const path &cam_path, Camera &cam);

bool ShowDepthMap(const path &depth_path, const cv::Mat& depth, float depth_min, float depth_max);

bool ShowNormalMap(const path &normal_path, const cv::Mat &normal);

bool ShowWeakImage(const path &weak_path, const cv::Mat &weak);

bool ExportPointCloud(const path& point_cloud_path, std::vector<PointList>& pointcloud);

std::string ToFormatIndex(int index);

template <typename TYPE>
void RescaleMatToTargetSize(const cv::Mat &src, cv::Mat &dst, const cv::Size2i &target_size);

void RunFusion(const path &dense_folder, const std::vector<Problem> &problems);

struct cudaTextureObjects {
	cudaTextureObject_t images[MAX_IMAGES];
};

struct DataPassHelper {
	int width;
	int height;
	int ref_index;
	cudaTextureObjects *texture_objects_cuda;
	cudaTextureObjects *texture_depths_cuda;
	Camera *cameras_cuda;
	float4 *plane_hypotheses_cuda;
	curandState *rand_states_cuda;
	unsigned int *selected_views_cuda;
	short2 *neighbours_cuda;
	int *neighbours_map_cuda;
	uchar *weak_info_cuda;
	float *costs_cuda;
	PatchMatchParams *params;
	int2 debug_point;
	bool show_ncc_info;
	float4* fit_plane_hypotheses_cuda;
	uchar* weak_reliable_cuda;
	uchar *view_weight_cuda;
	short2 *weak_nearest_strong;
#ifdef DEBUG_COST_LINE
	float *weak_ncc_cost_cuda;
#endif // DEBUG_COST_LINE

};

class APD {
public:
	APD(const Problem &problem);
	~APD();

	void InuputInitialization();
	void CudaSpaceInitialization();
	void SetDataPassHelperInCuda();
	void RunPatchMatch();
	float4 GetPlaneHypothesis(int r, int c);
	cv::Mat GetPixelStates();
	cv::Mat GetSelectedViews();
	int GetWidth();
	int GetHeight();
	float GetDepthMin();
	float GetDepthMax();
private:
	void GenerateWeakFromImage();

	int num_images;
	int width;
	int height;
	Problem problem;
	// =========================
	// image host and cuda
	std::vector<cv::Mat> images;
	cudaTextureObjects texture_objects_host;
	cudaArray *cuArray[MAX_IMAGES];
	cudaTextureObjects *texture_objects_cuda;
	// =========================
	// depth host and cuda
	std::vector<cv::Mat> depths;
	cudaTextureObjects texture_depths_host;
	cudaArray *cuDepthArray[MAX_IMAGES];
	cudaTextureObjects *texture_depths_cuda;
	// =========================
	// camera host and cuda
	std::vector<Camera> cameras;
	Camera *cameras_cuda;
	// =========================
	// weak info host and cuda
	int weak_count;
	cv::Mat weak_info_host;
	uchar *weak_info_cuda;
	uchar *weak_reliable_cuda;
	short2 *weak_nearest_strong;
	// =========================
	// neighbour host and cuda
	short2 *neighbours_cuda;
	cv::Mat neighbours_map_host;
	int *neigbours_map_cuda;
	// =========================
	// plane hypotheses host and cuda
	float4 *plane_hypotheses_host;
	float4 *plane_hypotheses_cuda;
	float4 *fit_plane_hypotheses_cuda;
	// =========================
	// cost cuda 
	float *costs_cuda;
	// =========================
	// other var
	// params
	PatchMatchParams params_host;
	PatchMatchParams *params_cuda;
	// random states
	curandState *rand_states_cuda;
	// vis info
	cv::Mat selected_views_host;
	unsigned int *selected_views_cuda;
	// for easy data pass
	DataPassHelper helper_host;
	DataPassHelper *helper_cuda;
	// save view weigth
	uchar *view_weight_cuda;
	//export for test
#ifdef DEBUG_COST_LINE
	float *weak_ncc_cost_cuda;
#endif // DEBUG_COST_LINE
};
#endif // !_APD_H_