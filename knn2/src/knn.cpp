// C++ program to find groups of unknown
// Points using K nearest neighbour algorithm.

#define STB_IMAGE_IMPLEMENTATION
   #include "stb_image.h"

#include <bits/stdc++.h>
#include <omp.h>
#include <time.h>
#include <cstdlib>


int n = 2; // Number of training images
// int n = (256*256); // Number of data points
#define TASK_SIZE 128

template<typename T>
void mergeSort(T *X, int n, T *tmp);
template<typename T>
void merge(T *X, int n, T *tmp) ;

template<typename T>
class ImageVec
{


public:
    int val;	                // Classification of Image
    double distance;            // distance
    T img[256*256];                    // image data type

 	ImageVec() {};
	~ImageVec() {};

    // get the distance of this object related to another object.
    double calcDistance(ImageVec * other_img)
    {
        double sum = 0.0;
        #pragma omp parallel for collapse(2) reduction(+: sum)
        for(unsigned int i = 0; i < 256; i++)
        {
            for(unsigned int j = 0; j < 256; j++)
            {
                double x = (double)(this->getPixel(i,j) - other_img->getPixel(i,j));
                sum += x * x;
            }
        }
        this->distance = std::sqrt(sum);
        return this->distance;
    }

    inline T getPixel(unsigned int i, unsigned int j)
    {
        return img[i * 256 + j];
    }

    inline bool operator<(ImageVec const& o_img)
    {
        return this->distance < o_img.distance;
    }

    static bool leq(ImageVec const& a, ImageVec const& b)
    {
        return a.distance < b.distance;
    }
};

// This function finds classification of point p using
// k nearest neighbour algorithm. It assumes only two
// groups and returns 0 if p belongs to group 0, else
// 1 (belongs to group 1).
template<typename T>
int classifyAPoint(T arr[], int n, int k, T * p)
{
	// Fill distances of all points from p
    // #pragma omp parallel for 
	for (int i = 0; i < n; i++){
        arr[i].calcDistance(p);
		// printf("Distance at %d was %lf\n",i,arr[i].distance);
    }
    
    T temp[n];
    // std::sort(arr, arr+n, T::leq);
    mergeSort(arr, n, temp);

	// Now consider the first k elements and only
	// two groups
	int freq1 = 0;	 // Frequency of group 0
	int freq2 = 0;	 // Frequency of group 1
    #pragma omp parallel for reduction(+:freq1, freq2)
	for (int i = 0; i < k; i++)
	{
        // printf("Distance at %d was %lf, val is %d\n",i,arr[i].distance, arr[i].val);
		if (arr[i].val == 0)
			freq1++;
		else if (arr[i].val == 1)
			freq2++;
	}

	return (freq1 > freq2 ? 0 : 1);
}

// Driver code
int main()
{
	ImageVec<uint8_t> arr[n];

    /* Define Test Data */
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            arr[0].img[i + 256 * j] = ( i + j) / 2;
        }
    }
    arr[0].val = 0;

    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            arr[1].img[i + 256 * j] = std::sqrt((128 - i) * (128 - i) + (128 - j)*(128 - j));
        }
    }

    arr[1].val = 1;

	/*Testing Point*/
	ImageVec<uint8_t> test;
	// for (int i = 0; i < 256; i++)
    // {
    //     for (int j = 0; j < 256; j++)
    //     {
    //         test.img[i + 256 * j] = ( i + j) / 3;
    //     }
    // }
     for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            test.img[i + 256 * j] = std::sqrt((100 - i) * (100 - i) + (100 - j)*(100 - j));
        }
    }

	// Parameter to decide group of the testing point
	int k = n/2;

    struct timespec begin;
    struct timespec end;
    double total_time = 0;
    uint64_t total_count = 0;
    for (int j = 1; j < 10; j = j * 2)
    {
        omp_set_num_threads(j);
        printf("Running with %d threads\n", j);

        int result;

        for (int l = 0; l < 10; l++)
        {
            clock_gettime(CLOCK_MONOTONIC, &begin);
            result = classifyAPoint(arr, n, k, &test);
            clock_gettime(CLOCK_MONOTONIC, &end);

            double elapsed = end.tv_sec - begin.tv_sec;
            elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
            total_time += elapsed;
            total_count ++;
        }

        printf ("Elapsed Time:%lf\n", total_time / total_count);
        printf ("The value classified to unknown point"
                " is %d.\n", result);
    }
   
	return 0;
}

template<typename T>
void merge(T *X, int n, T *tmp) {
   int i = 0;
   int j = n/2;
   int ti = 0;

   while (i<n/2 && j<n) {
      if (X[i] < X[j]) {
         tmp[ti] = X[i];
         ti++; i++;
      } else {
         tmp[ti] = X[j];
         ti++; j++;
      }
   }
   while (i<n/2) { /* finish up lower half */
      tmp[ti] = X[i];
      ti++; i++;
   }
   while (j<n) { /* finish up upper half */
      tmp[ti] = X[j];
      ti++; j++;
   }
   memcpy(X, tmp, n*sizeof(T));
} 

template<typename T>
void mergeSort(T *X, int n, T *tmp)
{
   if (n < 2) return;

   #pragma omp task shared(X) if (n > TASK_SIZE)
   mergeSort(X, n/2, tmp);

   #pragma omp task shared(X) if (n > TASK_SIZE)
   mergeSort(X+(n/2), n-(n/2), tmp + n/2);

   #pragma omp taskwait
   merge(X, n, tmp);
}

