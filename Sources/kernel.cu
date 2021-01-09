// smallptCUDA by Sam Lapere, 2015
// based on smallpt, a path tracer by Kevin Beason, 2008  
// Translated by Yang Kyowon, 2021

#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "cutil_math.cuh"// from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h

#define M_PI 3.14159265359f  // pi
#define width 512  // 화면 너비
#define height 384 // 화면 높이
#define samps 1024 // 픽셀당 샘플수

// __device__ : 디바이스(GPU)상에서 실행되며, 디바이스에서만 호출할 수 있습니다.
struct Ray 
{
   float3 orig; // ray origin
   float3 dir;  // ray direction 
   __device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // 재질(material) 타입들 입니다, radiance()에서 사용되고, 여기선 DIFF(Diffuse)만 사용됩니다.

struct Sphere
{
   float rad;            // 반지름
   float3 pos, emi, col; // 위치, 발광량(emission), 색상 
   Refl_t refl;          // 반사 타입 (e.g. diffuse<난반사>)

   __device__ float intersect_sphere(const Ray& r) const 
   {
      // 광선/구 교차
      // 교차 포인트로 거리 t를 반환합니다, 0을 반환한다면 교차하지 않은 것 입니다.
      // 광선 방정식: p(x, y, z) = ray.orig + t*ray.dir
      // 일반 구 방정식: x^2 + y^2 + z^2 = rad^2
      // ax^2 + bx + c = 0의 형태인 고전적인 2차 방정식
      // 위 2차방정식의 해 : x = (-b +- sqrt(b*b - 4ac)) / 2a
      // t^2*ray.dir*ray.dir + 2*t*(orig-p)*ray.dir + (orig-p)*(orig-p) - rad*rad = 0 의 해를 구하면 된다
      // more details in "Realistic Ray Tracing" book by P. Shirley or Scratchapixel.com

      float3 op = pos - r.orig;    // 구의 중심으로부터 ray.orign 까지의 거리(정확히는 변위)
      float t, epsilon = 0.0001f;  // 부동 소수점으로 인해 생기는 부동 소수점 정확도 오류(floating point precision artefacts)를 방지하기 위해 epsilon 값이 필요합니다
      float b = dot(op, r.dir);    // 이차 방정식의 b
      float disc = b * b - dot(op, op) + rad * rad;  // 이차 방정식의 판별식
      if (disc < 0) return 0;       // 만약 disc < 0 이면, 실근이 없다는 것 입니다(우리는 복소수해를 구할려는 것이 아니므로 무시합니다)
      else disc = sqrtf(disc);    // 만약 disc >= 0 이면, 음 그리고 양 판별식을 사용하여 해를 찾습니다
      return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0); // 광선의 원점의 앞부분과 가장 가까운 점을 선택합니다
   }
};

// SCENE
// 9 spheres forming a Cornell box
// small enough to be in constant GPU memory
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] = 
{
   { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF },   //Left 
   { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF },    //Rght 
   { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF },             //Back 
   { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF },//Frnt 
   { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF },             //Botm 
   { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF },    //Top 
   { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF },           // small sphere 1
   { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF },           // small sphere 2
   { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }   // Light
};

__device__ inline bool intersect_scene(const Ray& r, float& t, int& id) {

   float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;  // t는 가장 가까운 교차 거리 입니다, 그러므로 초기값으로 씬의 바깥쪽에 있을정도로 큰 값을 줍니다
   for (int i = int(n); i--;)  // 씬 안에 있는 모든 오브젝트 들에 대해 교차 검사를 합니다
   {
      if ((d = spheres[i].intersect_sphere(r)) && d < t)
      {  // 만약 지금보다 더 가까운(작은) 교차 거리가 계산된다면
         t = d;  // 광선을 따라 가장 가까운 교차점 까지의 거리와
         id = i; // 가장 가까운 교차 물체를 저장(추적)합니다
      }
   }

   return t < inf; // 씬에서 교차가 발생하면 true를 반환하고, 그렇지 않다면 false를 반환합니다
}

// random number generator from https://github.com/gz/rust-raytracer
__device__ static float getrandom(unsigned int* seed0, unsigned int* seed1) 
{
   *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
   *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

   unsigned int ires = ((*seed0) << 16) + (*seed1);

   // Convert to float
   union 
   {
      float f;
      unsigned int ui;
   } res;

   res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

   return (res.f - 2.f) / 2.f;
}

// Radiance 함수, 경로 추적(Path tracing)의 핵심
// Rendering Equation을 풀어내는 것은 즉:
// 나가는 radiance (한 점에서, outgoing radiance) = 방출(emitted)된 radiance + 반사된 radiance
// 반사된 radiance는 한 점위에 있는 반구내에서 모든 방향으로 부터 들어온 radiance(incoming radiance)의 합(적분)과,
// 재질의 반사 반사 함수(reflectance function, BRDF)와 입사각의 코사인의 곱을 구하는 것과 같습니다.
__device__ float3 radiance(Ray& r, unsigned int* s1, unsigned int* s2) // returns ray color
{ 
   float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // 바운스 루프를 통해 반복할 때 마다 광선의 색상을 축적합니다
   float3 mask = make_float3(1.0f, 1.0f, 1.0f);

   // ray bounce loop (no Russian Roulette used)
   // 광선 바운스 루프 (러시안 룰렛이 사용되지 않았음)
   for (int bounces = 0; bounces < 4; bounces++) // 최대 4번 튕길때 까지 반복 (CPU 코드에서의 재귀 호출을 대체합니다)
   {
      float t;           // 가장 가까운 교차점 까지의 거리
      int id = 0;        // 가장 가까운 교차 물체(구)의 인덱스

      // Scene안의 물체와 광선이 교차하는지 테스트 합니다
      if (!intersect_scene(r, t, id))
      {
         return make_float3(0.0f, 0.0f, 0.0f); // 만약 교차하지 않았다면 검은색(RGB(0, 0, 0))을 반환합니다.
      }
      
      // 만약, 교차한다는 것을 알아 냈다면
      // 교차점(hitpoint)과 노말을 계산합니다
      const Sphere& obj = spheres[id];             // 교차 오브젝트
      float3 x = r.orig + r.dir * t;               // 교차점
      float3 n = normalize(x - obj.pos);           // 노말 벡터(법선)
      float3 nl = dot(n, r.dir) < 0 ? n : n * -1;  // 물체의 바깥으로 향하는 노말 벡터(front facing normal)를 계산

      // 현재 구의 발광량을 축적된 색(accumulated colour)에 더합니다
      // (Rendering eqauation 합중 첫번째 항)
      accucolor += mask * obj.emi;

      // 씬안에 있는 모든 구체는 diffuse 재질 입니다
      // Diffuse 재질은 빛을 모든 방향으로 균등하게 반사합니다
      // 새로운 난반사 광선을 생성하기 위해선
      // 이전 광선의 경로 위에 있는 교차점(hitpoint)을 origin(출발 지점)으로 하고
      // 교차점 위의 반구에서 랜덤한 방향을 골라 그 방향을 새로운 광선의 방향으로 설정합니다. (see "Realistic Ray Tracing", P. Shirley)

      // 2개의 난수를 생성합니다
      float r1 = 2 * M_PI * getrandom(s1, s2);  // 방위각(azimuth)으로 사용하기 위한 단위원(반지름 = 1, 원 둘레 = 2*Pi인)에서 난수를 뽑습니다
      float r2 = getrandom(s1, s2);             // 고도각(elevation)으로 사용하기 위한 난수를 뽑습니다
      float r2s = sqrtf(r2);

      // 랜덤한 광선 방향을 계산에 사용하기 위한 교차점에서의 지역(local) 정규직교기저 uvw를 계산합니다
      // 이를 위해 첫 벡터로 교차점에서의 노말 벡터를 사용합니다(w), 그리고 두번째로는 이 첫 번째 벡터에 직교하는 벡터를 사용합니다(u), 
      // 마지막으로 세번째 벡터는 앞의 두 벡터에 직교하는 벡터를 사용합니다(v).
      float3 w = nl;
      float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
      float3 v = cross(w, u);

      // 극 좌표계를 이용해 반구상의 랜덤한 광선 방향을 계산해냅니다
      // cosine 가중치 importance sampling (광선의 방향을 노말 벡터의 방향에 가깝게 해줍니다)
      float3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

      // 새로운 광선의 원점은 이전 광선이 씬과 교차한 지점으로 설정합니다
      r.orig = x + nl * 0.05f; // 자가 교차를 피하기 위해 광선의 원점을 노말 벡터 방향으로 살짝 밀어줍니다
      r.dir = d;

      mask *= obj.col;     // 오브젝트의 색상과 곱합니다
      mask *= dot(d, nl);  // 입사광과 노말 벡터 사이의 각도를 사용하여 빛의 기여도에 가중치를 줍니다
      mask *= 2;           // 보정 계수(fudge factor)
   }

   return accucolor;
}


// __global__ : 디바이스(GPU)에서 실행되고, 호스트(CPU)에서만 호출 가능합니다
// 이 커널은 모든 쿠다 스레드들에서 병렬로 실행됩니다
__global__ void render_kernel(float3* output) {

   // CUDA 스레드를 모든 픽셀 (x, y)에 할당해줍니다
   // blockIdx, blockDim 그리고 threadIdx는 쿠다의 특별한 예약어(specific keywords) 입니다
   // CPU 코드에서 이미지의 행과 열을 순회하는 반복문을 대체합니다
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

   unsigned int i = (height - y - 1) * width + x; // 현재 픽셀의 인덱스 (thread 인덱스를 이용하여 계산됩니다)

   unsigned int s1 = x;  // 난수 생성기를 위한 시드값
   unsigned int s2 = y;

   // 화면의 왼쪽 코너를 가르키는 광선을 생성한후
   // 여기에 cx와 cy를 증가시키며 x 그리고 y 방향에 각각 더해주어 다른 모든 광선에 대한 방향들을 계산합니다
   Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // 하드 코딩된 첫 카메라 광선
   float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // 광선의 x 방향에 대한 오프셋
   float3 cy = normalize(cross(cx, cam.dir)) * .5135; // 광선이 y 방향에 대한 오프셋 (.5135는 FOV 각도 입니다; 약 30도(degree))
   float3 r; // 최종 픽셀 색상

   r = make_float3(0.0f); // r은 매 픽셀마다 0으로 초기화 됩니다

   for (int s = 0; s < samps; s++) // 픽셀당 샘플들
   {
      // 초기(primary) 광선의 방향을 계산합니다
      float3 d = cam.dir + cx * ((.25 + x) / width - .5) + cy * ((.25 + y) / height - .5);

      // 초기 광선을 생성하고, 들어오는 radiance 값을 pixelcolor에 더해줍니다.
      r = r + radiance(Ray(cam.orig + d * 40, normalize(d)), &s1, &s2) * (1. / samps);
   }       // Camera rays are pushed ^^^^^ forward to start in interior ; 카메라 광선이 내부에서 시작될 수 있도록 앞으로 밀어줍니다.


   // 픽셀의 RGB 값을 GPU에 있는 GPU에 적고, [0.0f, 1.0f] 범위 내로 자릅니다(clamp)
   output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

// [0.0, 1.0] 범위에 들어있는 부동소수점 RGB값을 [0, 255] 범위의 int 값으로 변환하고, gamma correction을 적용합니다
inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

int main() 
{
   float3* output_h = new float3[width * height]; // Host(System(Main) Memory)에 있는 이미지를 위한 메모리 공간을 가르키는 포인터
   float3* output_d;                             // Device(GPU VRAM)에 있는 이미지를 위한 메모리 공간을 가르키는 포인터

   // CUDA device에 메모리를 할당(GPU VRAM)
   cudaMalloc(&output_d, width * height * sizeof(float3));

   // dim3는 CUDA의 특별한 타입중 하나이며, block과 그리드는 SM(Streaming Multiprocessor)들에 CUDA 스레드들을 스케쥴링하기 위해 필요합니다
   dim3 block(8, 8, 1);
   dim3 grid(width / block.x, height / block.y, 1);

   printf("CUDA initialised.\nStart rendering...\n");

   // 스레드들을 디바이스에 스케쥴링하고 호스트로부터 CUDA 커널을 실행 합니다.
   render_kernel <<< grid, block >>> (output_d);

   // 계산 결과들을 디바이스(GPU Memory)로부터 호스트(System Memory)로 복사해옵니다.
   cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

   // CUDA 메모리(GPU Memory) 할당을 해제합니다
   cudaFree(output_d);

   printf("Done!\n");

   // 아주 간단한 이미지 포맷인 PPM을 사용하여 이미지를 저장합니다
   FILE* f = fopen("smallptcuda.ppm", "w");
   fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
   for (int i = 0; i < width * height; i++)  // 픽셀들을 돌아가며 RGB 값을 파일에 씁니다
   {
      fprintf(f, "%d %d %d ", 
         toInt(output_h[i].x),
         toInt(output_h[i].y),
         toInt(output_h[i].z));
   }

   printf("Saved image to 'smallptcuda.ppm'\n");
   delete[] output_h;
   //system("PAUSE");
}