extern bool iti_heiretu;
extern bool id_heiretu;
extern bool use_gpu;
extern bool camera_heiretu;
extern bool flir_camera;
extern bool idou;
extern std::ofstream zikken_output;
extern vector<vector<vector<Point2f>>> before_location_points;

extern vector<PostitPoint> before_postits_points;
extern vector<PostitPoint> postit_points;
extern vector<Mat> analyzing_images;

void printZikkenState(ofstream & out);

void realtimeDemoCategory(int argc, char * argv[]);

int main(int argc, char * argv[]);
