#include "APD.h"

bool ReadBinMat(const path &mat_path, cv::Mat &mat)
{
	ifstream in(mat_path, std::ios_base::binary);
	if (in.bad()) {
		std::cerr << "Error opening file: " << mat_path << std::endl;
		return false;
	}

	int version, rows, cols, type;
	in.read((char *)(&version), sizeof(int));
	in.read((char *)(&rows), sizeof(int));
	in.read((char *)(&cols), sizeof(int));
	in.read((char *)(&type), sizeof(int));

	if (version != 1) {
		in.close();
		std::cerr << "Version error: " << mat_path << std::endl;
		return false;
	}

	mat = cv::Mat(rows, cols, type);
	in.read((char *)mat.data, sizeof(char) * mat.step * mat.rows);
	in.close();
	return true;

}

bool WriteBinMat(const path &mat_path, const cv::Mat &mat) {

	ofstream out(mat_path, std::ios_base::binary);
	if (out.bad()) {
		std::cout << "Error opening file: " << mat_path << std::endl;
		return false;
	}
	int version = 1;
	int rows = mat.rows;
	int cols = mat.cols;
	int type = mat.type();

	out.write((char *)&version, sizeof(int));
	out.write((char *)&rows, sizeof(int));
	out.write((char *)&cols, sizeof(int));
	out.write((char *)&type, sizeof(int));
	out.write((char *)mat.data, sizeof(char) * mat.step * mat.rows);
	out.close();
	return true;
}

bool ReadCamera(const path &cam_path, Camera &cam)
{
	ifstream in(cam_path);
	if (in.bad()) {
		return false;
	}

	std::string line;
	in >> line;

	for (int i = 0; i < 3; ++i) {
		in >> cam.R[3 * i + 0] >> cam.R[3 * i + 1] >> cam.R[3 * i + 2] >> cam.t[i];
	}

	float tmp[4];
	in >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
	in >> line;

	for (int i = 0; i < 3; ++i) {
		in >> cam.K[3 * i + 0] >> cam.K[3 * i + 1] >> cam.K[3 * i + 2];
	}
	// compute camera center in world coord
	const auto &R = cam.R;
	const auto &t = cam.t;
	for (int j = 0; j < 3; ++j) {
		cam.c[j] = -float(double(R[0 + j])*double(t[0]) + double(R[3 + j])*double(t[1]) + double(R[6 + j])*double(t[2]));
	}
	// ====================================================================
	// TAT & ETH version read
	float depth_num;
	float interval;
	in >> cam.depth_min >> interval >> depth_num >> cam.depth_max;
	// ====================================================================
	////DTU version read
	//float depth_num = 192;
	//float interval;
	//in >> cam.depth_min >> interval;
	//cam.depth_max = interval * depth_num + cam.depth_min;
	////====================================================================
	in.close();
	return true;;
}

bool ShowDepthMap(const path &depth_path, const cv::Mat& depth, float depth_min, float depth_max)
{
	const float deltaDepth = depth_max - depth_min;
	// save image
	cv::Mat result_img(depth.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < depth.cols; i++) {
		for (int j = 0; j < depth.rows; j++) {
			if (depth.at<float>(j, i) < depth_min || depth.at<float>(j, i) > depth_max || isnan(depth.at<float>(j, i))) {
				continue;
			}
			float pixel_val = (depth_max - depth.at<float>(j, i)) / deltaDepth;
			if (pixel_val > 1) {
				pixel_val = 1;
			}
			if (pixel_val < 0) {
				pixel_val = 0;
			}
			pixel_val = pixel_val * 255;
			if (pixel_val>255) {
				pixel_val = 255;
			}
			else if (pixel_val<0) {
				pixel_val = 0;
			}
			auto &pixel = result_img.at<cv::Vec3b>(j, i);
			if (pixel_val <= 51)
			{
				pixel[0] = 255;
				pixel[1] = pixel_val * 5;
				pixel[2] = 0;
			}
			else if (pixel_val <= 102)
			{
				pixel_val -= 51;
				pixel[0] = 255 - pixel_val * 5;
				pixel[1] = 255;
				pixel[2] = 0;
			}
			else if (pixel_val <= 153)
			{
				pixel_val -= 102;
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = pixel_val * 5;
			}
			else if (pixel_val <= 204)
			{
				pixel_val -= 153;
				pixel[0] = 0;
				pixel[1] = 255 - static_cast<unsigned char>(pixel_val * 128.0 / 51 + 0.5);
				pixel[2] = 255;
			}
			else if (pixel_val <= 255)
			{
				pixel_val -= 204;
				pixel[0] = 0;
				pixel[1] = 127 - static_cast<unsigned char>(pixel_val * 127.0 / 51 + 0.5);
				pixel[2] = 255;
			}
			
		}
	}
	cv::imwrite(depth_path.string(), result_img);
	return true;
}

bool ShowNormalMap(const path &normal_path, const cv::Mat &normal)
{
	if (normal.empty()) {
		return false;
	}
	cv::Mat normalized_normal = normal.clone();
	for (int i = 0; i < normalized_normal.rows; i++) {
		for (int j = 0; j < normalized_normal.cols; j++) {
			cv::Vec3f normal_val = normalized_normal.at<cv::Vec3f>(i, j);
			float norm = sqrt(pow(normal_val[0], 2) + pow(normal_val[1], 2) + pow(normal_val[2], 2));
			if (norm == 0) {
				normalized_normal.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
			}
			else {
				normalized_normal.at<cv::Vec3f>(i, j) = normal_val / norm;
			}
		}
	}

	cv::Mat img(normalized_normal.size(), CV_8UC3, cv::Scalar(0.f, 0.f, 0.f));
	normalized_normal.convertTo(img, img.type(), 255.f / 2.f, 255.f / 2.f);
	cv::imwrite(normal_path.string(), img);
	return true;
}

bool ShowWeakImage(const path &weak_path, const cv::Mat &weak) {
	// show image
	if (weak.empty()) {
		return false;
	}
	const int width = weak.cols;
	const int height = weak.rows;
	cv::Mat weak_info_image(height, width, CV_8UC3);
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			switch (weak.at<uchar>(r, c))
			{
			case WEAK:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
				break;
			case STRONG:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0);
				break;
			case UNKNOWN:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255);
				break;
			}
		}
	}
	// save
	cv::imwrite(weak_path.string(), weak_info_image);
	return true;
}

bool ExportPointCloud(const path& point_cloud_path, std::vector<PointList>& pointcloud)
{
	ofstream out(point_cloud_path, std::ios::binary);
	if (out.bad()) {
		return false;
	}

	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "element vertex " << int(pointcloud.size()) << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property uchar diffuse_blue\n";
	out << "property uchar diffuse_green\n";
	out << "property uchar diffuse_red\n";
	out << "end_header\n";

	for (size_t idx = 0; idx < pointcloud.size(); idx++)
	{
		float px = pointcloud[idx].coord.x;
		float py = pointcloud[idx].coord.y;
		float pz = pointcloud[idx].coord.z;


		cv::Vec3b pixel;
		pixel[0] = static_cast<uchar>(pointcloud[idx].color.x);
		pixel[1] = static_cast<uchar>(pointcloud[idx].color.y);
		pixel[2] = static_cast<uchar>(pointcloud[idx].color.z);

		out.write((char *)&px, sizeof(float));
		out.write((char *)&py, sizeof(float));
		out.write((char *)&pz, sizeof(float));

		out.write((char *)&pixel[0], sizeof(uchar));
		out.write((char *)&pixel[1], sizeof(uchar));
		out.write((char *)&pixel[2], sizeof(uchar));
	}
	out.close();
	return true;
}

void StringAppendV(std::string* dst, const char* format, va_list ap) {
	// First try with a small fixed size buffer.
	static const int kFixedBufferSize = 1024;
	char fixed_buffer[kFixedBufferSize];

	// It is possible for methods that use a va_list to invalidate
	// the data in it upon use.  The fix is to make a copy
	// of the structure before using it and use that copy instead.
	va_list backup_ap;
	va_copy(backup_ap, ap);
	int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
	va_end(backup_ap);

	if (result < kFixedBufferSize) {
		if (result >= 0) {
			// Normal case - everything fits.
			dst->append(fixed_buffer, result);
			return;
		}

#ifdef _MSC_VER
		// Error or MSVC running out of space.  MSVC 8.0 and higher
		// can be asked about space needed with the special idiom below:
		va_copy(backup_ap, ap);
		result = vsnprintf(nullptr, 0, format, backup_ap);
		va_end(backup_ap);
#endif

		if (result < 0) {
			// Just an error.
			return;
		}
	}

	// Increase the buffer size to the size requested by vsnprintf,
	// plus one for the closing \0.
	const int variable_buffer_size = result + 1;
	std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

	// Restore the va_list before we use it again.
	va_copy(backup_ap, ap);
	result =
		vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
	va_end(backup_ap);

	if (result >= 0 && result < variable_buffer_size) {
		dst->append(variable_buffer.get(), result);
	}
}

std::string StringPrintf(const char* format, ...) {
	va_list ap;
	va_start(ap, format);
	std::string result;
	StringAppendV(&result, format, ap);
	va_end(ap);
	return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
	const int line) {
	if (error != cudaSuccess) {
		std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
			file.c_str(), line)
			<< std::endl;
		exit(EXIT_FAILURE);
	}
}

void CudaCheckError(const char* file, const int line) {
	cudaError error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
			line, cudaGetErrorString(error))
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	error = cudaDeviceSynchronize();
	if (cudaSuccess != error) {
		std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
			file, line, cudaGetErrorString(error))
			<< std::endl;
		std::cerr
			<< "This error is likely caused by the graphics card timeout "
			"detection mechanism of your operating system. Please refer to "
			"the FAQ in the documentation on how to solve this problem."
			<< std::endl;
		exit(EXIT_FAILURE);
	}
}

std::string ToFormatIndex(int index) {
	std::stringstream ss;
	ss << std::setw(8) << std::setfill('0') << index;
	return ss.str();
}

APD::APD(const Problem &problem) {
	params_host = problem.params;
	this->problem = problem;
}

APD::~APD() {
	delete[] plane_hypotheses_host;
	// free images
	{
		for (int i = 0; i < num_images; ++i) {
			cudaDestroyTextureObject(texture_objects_host.images[i]);
			cudaFreeArray(cuArray[i]);
		}
		cudaFree(texture_objects_cuda);
	}
	// may free depths
	if (params_host.geom_consistency) {
		for (int i = 0; i < num_images; ++i) {
			cudaDestroyTextureObject(texture_depths_host.images[i]);
			cudaFreeArray(cuDepthArray[i]);
		}
		cudaFree(texture_depths_cuda);
	}
	cudaFree(cameras_cuda);
	cudaFree(plane_hypotheses_cuda);
	cudaFree(fit_plane_hypotheses_cuda);
	cudaFree(costs_cuda);
	cudaFree(rand_states_cuda);
	cudaFree(selected_views_cuda);
	cudaFree(params_cuda);
	cudaFree(helper_cuda);
	cudaFree(neighbours_cuda);
	cudaFree(neigbours_map_cuda);
	cudaFree(weak_info_cuda);
	cudaFree(weak_reliable_cuda);
	cudaFree(view_weight_cuda);
	cudaFree(weak_nearest_strong);
#ifdef DEBUG_COST_LINE
	cudaFree(weak_ncc_cost_cuda);
#endif // DEBUG_COST_LINE

}

void APD::InuputInitialization() {
	images.clear();
	cameras.clear();
	// get folder
	path image_folder = problem.dense_folder / path("images");
	path cam_folder = problem.dense_folder / path("cams");
	//path weak_folder = problem.dense_folder / path("weaks");
	// =================================================
	// read ref image and src images
	// ref
	{ 
		path ref_image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		cv::Mat_<uint8_t> image_uint = cv::imread(ref_image_path.string(), cv::IMREAD_GRAYSCALE);
		cv::Mat image_float;
		image_uint.convertTo(image_float, CV_32FC1);
		images.push_back(image_float);
		width = image_float.cols;
		height = image_float.rows;
	}
	// src
	for (const auto &src_idx : problem.src_image_ids) {
		path src_image_path = image_folder / path(ToFormatIndex(src_idx) + ".jpg");
		cv::Mat_<uint8_t> image_uint = cv::imread(src_image_path.string(), cv::IMREAD_GRAYSCALE);
		cv::Mat image_float;
		image_uint.convertTo(image_float, CV_32FC1);
		images.push_back(image_float);
		// assert: images_float.cols == width;
		// assert: images_float.rows == height;
	}
	if (images.size() > MAX_IMAGES) {
		std::cerr << "Can't process so much images: " << images.size() << std::endl;
		exit(EXIT_FAILURE);
	}
	// =================================================
	// read ref camera and src camera
	// ref
	{
		path ref_cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera cam;
		ReadCamera(ref_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// src
	for (const auto &src_idx : problem.src_image_ids) {
		path src_cam_path = cam_folder / path(ToFormatIndex(src_idx) + "_cam.txt");
		Camera cam;
		ReadCamera(src_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// =================================================
	// set some params
	params_host.depth_min = cameras[0].depth_min * 0.6f;
	params_host.depth_max = cameras[0].depth_max * 1.2f;
	params_host.num_images = (int)images.size();
	num_images = (int)images.size();
	// =================================================
	std::cout << "Read images and camera done\n";
	std::cout << "Depth range: " << params_host.depth_min << " " << params_host.depth_max << std::endl;
	std::cout << "Num images: " << params_host.num_images << std::endl;
	// =================================================
	// scale images
	if (problem.scale_size != 1) {
		for (int i = 0; i < num_images; ++i) {
			const float factor = 1.0f / (float)(problem.scale_size);
			const int new_cols = std::round(images[i].cols * factor);
			const int new_rows = std::round(images[i].rows * factor);

			const float scale_x = new_cols / static_cast<float>(images[i].cols);
			const float scale_y = new_rows / static_cast<float>(images[i].rows);

			cv::Mat_<float> scaled_image_float;
			cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
			images[i] = scaled_image_float.clone();

			width = scaled_image_float.cols;
			height = scaled_image_float.rows;

			cameras[i].K[0] *= scale_x;
			cameras[i].K[2] *= scale_x;
			cameras[i].K[4] *= scale_y;
			cameras[i].K[5] *= scale_y;
			cameras[i].width = width;
			cameras[i].height = height;
		}
		std::cout << "Scale images and cameras done\n";
	}
	std::cout << "Image size: " << width << " * " << height << std::endl;
	// =================================================
	// read depth form geom consistency
	if (params_host.geom_consistency) {
		depths.clear();
		path ref_depth_path = problem.result_folder / path("depths.dmb");
		cv::Mat ref_depth;
		ReadBinMat(ref_depth_path, ref_depth);
		depths.push_back(ref_depth);
		for (const auto &src_idx : problem.src_image_ids) {
			path src_depth_path = problem.dense_folder / path("APD") / path(ToFormatIndex(src_idx)) / path("depths.dmb");
			cv::Mat src_depth;
			ReadBinMat(src_depth_path, src_depth);
			depths.push_back(src_depth);
		}
		for (auto &depth : depths) {
			if (depth.cols != width || depth.rows != height) {
				RescaleMatToTargetSize<float>(depth, depth, cv::Size(width, height));
			}
		}

	}
	// =================================================
	// read weak info
	if (params_host.use_APD) {
		path weak_info_path = problem.result_folder / path("weak.bin");
		if (!exists(weak_info_path)) {
			std::cerr << "Can't find weak info file: " << weak_info_path.string() << std::endl;
			exit(EXIT_FAILURE);
		}
		ReadBinMat(weak_info_path, weak_info_host);
		if (weak_info_host.cols != width || weak_info_host.rows != height) {
			std::cerr << "Weak info doesn't match the images' size!\n";
			RescaleMatToTargetSize<uchar>(weak_info_host, weak_info_host, cv::Size(width, height));
			std::cout << "Scale done\n";
		}
		
		neighbours_map_host = cv::Mat::zeros(weak_info_host.size(), CV_32SC1);
		weak_count = 0;
		for (int r = 0; r < weak_info_host.rows; ++r) {
			for (int c = 0; c < weak_info_host.cols; ++c) {
				int val = weak_info_host.at<uchar>(r, c);
				// point is weak
				if (val == WEAK) {
					neighbours_map_host.at<int>(r, c) = weak_count;
					weak_count++;
				}
			}
		}
		std::cout << "Weak count: " << weak_count << " / " << weak_info_host.cols * weak_info_host.rows << " = " << (float)weak_count / (float)(weak_info_host.cols * weak_info_host.rows) * 100 << "%" << std::endl;
	}
	else {
		weak_info_host = cv::Mat::zeros(height, width, CV_8UC1);
		weak_count = 0;
		for (int r = 0; r < weak_info_host.rows; ++r) {
			for (int c = 0; c < weak_info_host.cols; ++c) {
				weak_info_host.at<uchar>(r, c) = STRONG;
			}
		}
	}
	// =================================================
	plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
	selected_views_host = cv::Mat::zeros(height, width, CV_32SC1);
	if (params_host.state != FIRST_INIT) {
		// input plane hypotheses from existed result
		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);
		if (depth.cols != width || depth.rows != height || normal.cols != width || normal.rows != height) {
			std::cerr << "Depth and Normal doesn't match the images' size!\n";
			RescaleMatToTargetSize<float>(depth, depth, cv::Size2i(width, height));
			RescaleMatToTargetSize<cv::Vec3f>(normal, normal, cv::Size2i(width, height));
		}
		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				plane_hypotheses_host[center].w = depth.at<float>(row, col);
				plane_hypotheses_host[center].x = normal.at<cv::Vec3f>(row, col)[0];
				plane_hypotheses_host[center].y = normal.at<cv::Vec3f>(row, col)[1];
				plane_hypotheses_host[center].z = normal.at<cv::Vec3f>(row, col)[2];
			}
		}
		{
			path selected_view_path = problem.result_folder / path("selected_views.bin");
			ReadBinMat(selected_view_path, selected_views_host);
			if (selected_views_host.cols != width || selected_views_host.rows != height) {
				std::cerr << "Select view doesn't match the images' size!\n";
				RescaleMatToTargetSize<unsigned int>(selected_views_host, selected_views_host, cv::Size2i(width, height));
			}
		}
	}
	// =================================================
}

void APD::CudaSpaceInitialization() {
	// =================================================
	// move images to gpu
	for (int i = 0; i < num_images; ++i) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaMallocArray(&cuArray[i], &channelDesc, width, height);
		cudaMemcpy2DToArray(cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray[i];
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;
		cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
	}
	cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
	cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
	// may move depths to gpu
	if (params_host.geom_consistency) {
		for (int i = 0; i < num_images; ++i) {
			int height = depths[i].rows;
			int width = depths[i].cols;
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			cudaMallocArray(&cuDepthArray[i], &channelDesc, width, height);
			cudaMemcpy2DToArray(cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(cudaResourceDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cuDepthArray[i];
			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(cudaTextureDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 0;
			cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
		}
		cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
		cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
	}
	// =================================================
	// move camera to gpu
	cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
	cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);
	// malloc memory for important data structure
	const int length = width * height;
	// define cost
	cudaMalloc((void**)&costs_cuda, sizeof(float) * length);
	// malloc memory for rand states
	cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * length);
	// malloc for selected_views
	cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * length);
	cudaMemcpy(selected_views_cuda, selected_views_host.ptr<unsigned int>(0), sizeof(unsigned int) * length, cudaMemcpyHostToDevice);
	// view weight
	cudaMalloc((void**)&view_weight_cuda, sizeof(uchar) * length * MAX_IMAGES);
	// move plane hypotheses to gpu
	cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * length);
	cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * length, cudaMemcpyHostToDevice);
	// malloc memory for fit plane 
	cudaMalloc((void**)&fit_plane_hypotheses_cuda, sizeof(float4) * length);
	cudaMemset(fit_plane_hypotheses_cuda, 0, sizeof(float4) * length);
	// malloc memory for weak info
	cudaMalloc((void **)(&weak_info_cuda), length * sizeof(uchar));
	cudaMemcpy(weak_info_cuda, weak_info_host.ptr<uchar>(0), length * sizeof(uchar), cudaMemcpyHostToDevice);
	// malloc memory for weak reliable info
	cudaMalloc((void **)(&weak_reliable_cuda), length * sizeof(uchar));
	// malloc memory for nearest strong points
	cudaMalloc((void**)(&weak_nearest_strong), length * sizeof(short2));
	// move neighbour map to gpu
	cudaMalloc((void**)(&neigbours_map_cuda), length * sizeof(int));
	cudaMemcpy(neigbours_map_cuda, neighbours_map_host.ptr<int>(0), length * sizeof(int), cudaMemcpyHostToDevice);
	// malloc memory for deformable ncc
	cudaMalloc((void **)(&neighbours_cuda), weak_count * NEIGHBOUR_NUM * sizeof(short2));
	// move param to gpu
	cudaMalloc((void**)(&params_cuda), sizeof(PatchMatchParams));
	cudaMemcpy(params_cuda, &params_host, sizeof(PatchMatchParams), cudaMemcpyHostToDevice);
	// =================================================
#ifdef DEBUG_COST_LINE
	cudaMalloc((void**)(&weak_ncc_cost_cuda), sizeof(float) * width * height * 61);
#endif // DEBUG_COST_LINE
}

void APD::SetDataPassHelperInCuda() {
	helper_host.width = this->width;
	helper_host.height = this->height;
	helper_host.ref_index = this->problem.ref_image_id;
	helper_host.texture_depths_cuda = this->texture_depths_cuda;
	helper_host.texture_objects_cuda = this->texture_objects_cuda;
	helper_host.cameras_cuda = this->cameras_cuda;
	helper_host.costs_cuda = this->costs_cuda;
	helper_host.neighbours_cuda = this->neighbours_cuda;
	helper_host.neighbours_map_cuda = this->neigbours_map_cuda;
	helper_host.plane_hypotheses_cuda = this->plane_hypotheses_cuda;
	helper_host.rand_states_cuda = this->rand_states_cuda;
	helper_host.selected_views_cuda = this->selected_views_cuda;
	helper_host.weak_info_cuda = this->weak_info_cuda;
	helper_host.params = params_cuda;
	helper_host.debug_point = make_int2(DEBUG_POINT_X, DEBUG_POINT_Y);
	helper_host.show_ncc_info = false;
	helper_host.fit_plane_hypotheses_cuda = fit_plane_hypotheses_cuda;
	helper_host.weak_reliable_cuda = weak_reliable_cuda;
	helper_host.view_weight_cuda = view_weight_cuda;
	helper_host.weak_nearest_strong = weak_nearest_strong;
#ifdef DEBUG_COST_LINE
	helper_host.weak_ncc_cost_cuda = weak_ncc_cost_cuda;
#endif // DEBUG_COST_LINE
	cudaMalloc((void**)(&helper_cuda), sizeof(DataPassHelper));
	cudaMemcpy(helper_cuda, &helper_host, sizeof(DataPassHelper), cudaMemcpyHostToDevice);
}

float4 APD::GetPlaneHypothesis(int r, int c) {
	return plane_hypotheses_host[c + r * width];
}

cv::Mat APD::GetPixelStates() {
	return weak_info_host;
}

cv::Mat APD::GetSelectedViews() {
	return selected_views_host;
}

int APD::GetWidth() {
	return width;
}

int APD::GetHeight() {
	return height;
}

float APD::GetDepthMin() {
	return params_host.depth_min;
}

float APD::GetDepthMax() {
	return params_host.depth_max;
}

void RescaleImageAndCamera(cv::Mat &src, cv::Mat &dst, cv::Mat &depth, Camera &camera)
{
	const int cols = depth.cols;
	const int rows = depth.rows;

	if (cols == src.cols && rows == src.rows) {
		dst = src.clone();
		return;
	}

	const float scale_x = cols / static_cast<float>(src.cols);
	const float scale_y = rows / static_cast<float>(src.rows);

	cv::resize(src, dst, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);

	camera.K[0] *= scale_x;
	camera.K[2] *= scale_x;
	camera.K[4] *= scale_y;
	camera.K[5] *= scale_y;
	camera.width = cols;
	camera.height = rows;
}

template <typename TYPE>
void RescaleMatToTargetSize(const cv::Mat &src, cv::Mat &dst, const cv::Size2i &target_size) {
	if (src.cols == target_size.width && src.rows == target_size.height) {
		return;
	}
	const float scale_x = target_size.width / static_cast<float>(src.cols);
	const float scale_y = target_size.height / static_cast<float>(src.rows);

	int type = src.type();
	cv::Mat src_clone = src.clone();
	dst = cv::Mat(target_size.height, target_size.width, type);

	for (int r = 0; r < target_size.height; ++r) {
		for (int c = 0; c < target_size.width; ++c) {
			int o_r = static_cast<int>(r / scale_x);
			int o_c = static_cast<int>(c / scale_y);
			if (o_r < 0 || o_c < 0 || o_r >= src_clone.rows || o_c >= src_clone.cols) {
				continue;
			}
			dst.at<TYPE>(r, c) = src_clone.at<TYPE>(o_r, o_c);
		}
	}
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
	float3 pointX;
	float3 tmpX;
	// Reprojection
	pointX.x = depth * (x - camera.K[2]) / camera.K[0];
	pointX.y = depth * (y - camera.K[5]) / camera.K[4];
	pointX.z = depth;

	// Rotation
	tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
	tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
	tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

	// Transformation
	float3 C;
	C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
	C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
	C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
	pointX.x = tmpX.x + C.x;
	pointX.y = tmpX.y + C.y;
	pointX.z = tmpX.z + C.z;

	return pointX;
}

void ProjectCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
	float3 tmp;
	tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
	tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
	tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

	depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
	point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
	point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2)
{
	float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	float angle = acosf(dot_product);
	//if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
	if (angle != angle)
		return 0.0f;

	return angle;
}

// ETH version
void RunFusion(const path &dense_folder, const std::vector<Problem> &problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	std::vector<cv::Mat> weaks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	weaks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto &problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);
	
		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		path weak_path = problem.result_folder / path("weak.bin");
		cv::Mat depth, normal, weak;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);
		ReadBinMat(weak_path, weak);
	
		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			blocks.emplace_back(block_jpg);
		}
	
		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
		RescaleMatToTargetSize<uchar>(weak, weak, cv::Size2i(depth.cols, depth.rows));
		weaks.emplace_back(weak);
	}
	std::vector<PointList> PointCloud;
	PointCloud.clear();

	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto &problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				if (masks[ref_index].at<uchar>(r, c) == 1) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				int num_consistent = 0;
				float dynamic_consistency = 0.0f;
				std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;
							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}
				float factor = (weaks[ref_index].at<uchar>(r, c) == WEAK ? 0.45f : 0.3f);
				if (num_consistent >= 1 && (dynamic_consistency > factor * num_consistent)) {
					PointList point3D;
					point3D.coord = consistent_Point;
					float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };
					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						int src_index = imageIdToindexMap[problem.src_image_ids[j]];
						masks[src_index].at<uchar>(used_list[j].y, used_list[j].x) = 1;
						const auto &color = images[src_index].at<cv::Vec3b>(used_list[j].y, used_list[j].x);
						consistent_Color[0] += color[0];
						consistent_Color[1] += color[1];
						consistent_Color[2] += color[2];
					}
					consistent_Color[0] /= (num_consistent + 1);
					consistent_Color[1] /= (num_consistent + 1);
					consistent_Color[2] /= (num_consistent + 1);
					point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
					PointCloud.emplace_back(point3D);
				}

			}
		}
	}
	path ply_path = dense_folder / path("APD") / path("APD.ply");
	ExportPointCloud(ply_path, PointCloud);
}

void RunFusion_TAT_Intermediate(const path &dense_folder, const std::vector<Problem> &problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");
	const float dist_base = 0.25f;
	const float depth_base = 1.0f / 3500.0f;

	const float angle_base = 0.06981317007977318f; // 4 degree
	const float angle_grad = 0.05235987755982988f; // 3 degree

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto &problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);

		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);

		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			blocks.emplace_back(block_jpg);
		}

		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	struct CostData
	{
		float dist;
		float depth;
		float angle;
		int src_r;
		int src_c;
		bool use;

		CostData () {
			dist = FLT_MAX;
			depth = FLT_MAX;
			angle = FLT_MAX;
		}
	};
	

	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto &problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		std::vector<CostData> diff(num_ngb, CostData());
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);
						diff[j].dist = reproj_error;
						diff[j].depth = relative_depth_diff;
						diff[j].angle = angle;
						diff[j].src_r = src_r;
						diff[j].src_c = src_c;
					}
				}
				for (int k = 2; k <= num_ngb; ++k) {
					int count = 0;
					for (int j = 0; j < num_ngb; ++j) {
						diff[j].use = false;
						if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base && diff[j].angle < (k * angle_grad + angle_base)) {
							count++;
							diff[j].use = true;
						}
					}
					if (count >= k) {
						PointList point3D;						
						float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };
						for (int j = 0; j < num_ngb; ++j) {
							if (diff[j].use) {
								int src_index = imageIdToindexMap[problem.src_image_ids[j]];
								consistent_Color[0] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[0];
								consistent_Color[1] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[1];
								consistent_Color[2] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[2];
							}
						}
						consistent_Color[0] /= (count + 1.0f);
						consistent_Color[1] /= (count + 1.0f);
						consistent_Color[2] /= (count + 1.0f);

						point3D.coord = consistent_Point;
						point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
						PointCloud.emplace_back(point3D);
						masks[ref_index].at<uchar>(r, c) = 1;
						break;
					}
				}
			}
		}
	}
	path ply_path = dense_folder / path("APD") / path("APD.ply");
	ExportPointCloud(ply_path, PointCloud);
}

void RunFusion_TAT_advanced(const path &dense_folder, const std::vector<Problem> &problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");
	const float dist_base = 0.25f;
	const float depth_base = 1.0f / 3000.0f;

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto &problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);

		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);

		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			blocks.emplace_back(block_jpg);
		}

		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	struct CostData
	{
		float dist;
		float depth;
		float angle;

		CostData () {
			dist = FLT_MAX;
			depth = FLT_MAX;
			angle = FLT_MAX;
		}
	};
	

	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto &problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		std::vector<CostData> diff(num_ngb, CostData());
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };

				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);
						diff[j].dist = reproj_error;
						diff[j].depth = relative_depth_diff;
						diff[j].angle = angle;
					}
				}
				for (int k = 2; k <= num_ngb; ++k) {
					int count = 0;
					for (int j = 0; j < num_ngb; ++j) {
						if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base) {
							count++;
						}
					}
					if (count >= k) {
						PointList point3D;
						point3D.coord = consistent_Point;
						point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
						PointCloud.emplace_back(point3D);
						masks[ref_index].at<uchar>(r, c) = 1;
						break;
					}
				}
			}
		}
	}
	path ply_path = dense_folder / path("APD") / path("APD.ply");
	ExportPointCloud(ply_path, PointCloud);
}
