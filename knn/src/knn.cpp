// C++ program to find groups of unknown
// Points using K nearest neighbour algorithm.
#include <bits/stdc++.h>
#include <omp.h>
#include <time.h>
#include <cstdlib>

int n = (256*256); // Number of data points
#define TASK_SIZE 128

template<typename T>
void mergeSort(T *X, int n, T *tmp);
template<typename T>
void merge(T *X, int n, T *tmp) ;

// struct Point
// {
// 	int val;	 // Group of point
// 	double x, y;	 // Co-ordinate of point
// 	double distance; // Distance from test point
// };

class Point
{
public:
	int val;	 // Group of point
	double x, y;	 // Co-ordinate of point
	double distance; // Distance from test point

 	Point() {};
	~Point() {};

	// Used to sort an array of points by increasing
	// order of distance
	inline bool operator<(Point const& b) {
		return (this->distance < b.distance);
	};

	// Used to sort an array of points by increasing
	// order of distance
	inline bool operator==(Point const& b) {
		return (this->distance == b.distance);
	};

	// Used to sort an array of points by increasing
	// order of distance
	inline bool operator<=(Point const& b) {
		return (this->distance <= b.distance);
	};

	static bool leq(Point a, Point b)
	{
		return a < b;
	}
};

// This function finds classification of point p using
// k nearest neighbour algorithm. It assumes only two
// groups and returns 0 if p belongs to group 0, else
// 1 (belongs to group 1).
int classifyAPoint(Point arr[], int n, int k, Point p)
{
	// Fill distances of all points from p
    #pragma omp parallel for 
	for (int i = 0; i < n; i++)
		arr[i].distance =
			sqrt((arr[i].x - p.x) * (arr[i].x - p.x) +
				(arr[i].y - p.y) * (arr[i].y - p.y));
    
    Point temp[n];
    // std::sort(arr, arr+n, Point::leq);
    mergeSort(arr, n, temp);

	// Now consider the first k elements and only
	// two groups
	int freq1 = 0;	 // Frequency of group 0
	int freq2 = 0;	 // Frequency of group 1
    #pragma omp parallel for reduction(+:freq1, freq2)
	for (int i = 0; i < k; i++)
	{
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
	
	Point arr[n];

    for (int i = 0; i < n; i++)
    {
        arr[i].x = rand() % 1000;
        arr[i].y = rand() % 1000;
        arr[i].val = rand() % 2;
    }

	/*Testing Point*/
	Point p;
	p.x = 3;
	p.y = 10;

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

        for (int l = 0; l < 10; l++)
        {
            clock_gettime(CLOCK_MONOTONIC, &begin);
            int result = classifyAPoint(arr, n, k, p);
            clock_gettime(CLOCK_MONOTONIC, &end);

            double elapsed = end.tv_sec - begin.tv_sec;
            elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
            total_time += elapsed;
            total_count ++;
        }

        printf ("Elapsed Time:%lf\n", total_time / total_count);
        printf ("The value classified to unknown point"
                " is %d.\n", 1);
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
   memcpy(X, tmp, n*sizeof(Point));
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