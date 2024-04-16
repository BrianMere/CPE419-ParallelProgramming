#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "life.h"
#include "graphics.h"

const char pulsar[13][14] = {
"..OOO...OOO..",
".............",
"O....O.O....O",
"O....O.O....O",
"O....O.O....O",
"..OOO...OOO..",
".............",
"..OOO...OOO..",
"O....O.O....O",
"O....O.O....O",
"O....O.O....O",
".............",
"..OOO...OOO.."
};


uint8_t * arr1;
uint8_t * arr2;
uint8_t switch_flag = 0;

void init_arrays();
void seed_arrays();
void life_main();
uint8_t get_alive(uint8_t * array, uint32_t row, uint32_t col);

int main()
{
   
   int n = N;
   init_arrays();
   seed_arrays();

   while (1)
   {
      life_main();
      // if (getline())
      // {
      //    break;
      // }
      usleep(FRAME_DELAY);
   }

   free(arr1);
   free(arr2);
   return 0;
}

void life_main()
{
   uint8_t * past_array    = (switch_flag) ? arr1 : arr2;
   uint8_t * future_array  = (switch_flag) ? arr2 : arr1;
   switch_flag = !switch_flag;
   #pragma omp parallel for shared(past_array, future_array) default(none)
   for (int i = 0; i < N * N; i++)
   {
      uint32_t row = i / N;
      uint32_t col = i % N;
      uint8_t cell_status = get_cell(past_array, row, col);
      uint8_t neighbors = get_alive(past_array, row, col);

      // If a cell has less than 2 or more than 3 neighbors, kill it
      if (neighbors < 2 || neighbors > 3)
      {
         set_cell(future_array, row, col, DEAD);
      }
      // If an alive cell has 2 or 3 neighbors, it is alive
      else if (cell_status == ALIVE && (neighbors == 2 || neighbors == 3))
      {
         set_cell(future_array, row, col, ALIVE);
      }
      // If a dead cell has 3 neighbors, revive it
      else if (cell_status == DEAD && neighbors == 3)
      {
         set_cell(future_array, row, col, ALIVE);
      }
      // Otherwise, maintain its status
      else
      {
         set_cell(future_array, row, col, cell_status);
      }
   }

   print_screen(past_array, N);
}

void init_arrays () {
   arr1 = malloc(sizeof(uint8_t) * N * N);
   if (arr1 == NULL)
   {
      perror("malloc");
      exit(-1);
   }


   arr2 = malloc(sizeof(uint8_t) * N * N);
   if (arr2 == NULL)
   {
      perror("malloc");
      exit(-1);
   }

}

void seed_arrays()
{

   for (int row = 0; row < N; row++)
      for (int col = 0; col < N; col++)
         {
               set_cell(arr1, row, col, 0);
         }
   for (int row = 0; row < N; row++)
      for (int col = 0; col < N; col++)
         {
               set_cell(arr2, row, col, 0);
         }

   // Spinner
   for (int col = 0; col < 3; col++)
      {
            set_cell(arr2, 2, col, 1);
      }

   // Pulsar
   for (int row = 0; row < sizeof(pulsar) / (sizeof(pulsar[0])); row++)
   for (int col = 0; col < sizeof(pulsar[0]) - 1; col++)
      {
         if (pulsar[row][col] == '.')
            set_cell(arr2, row+PULSAR_OFFSET, col+PULSAR_OFFSET, 0);
         else
            {
               set_cell(arr2, row+PULSAR_OFFSET, col+PULSAR_OFFSET, 1);
            }
      }


   // Random
   // for (int row = 0; row < N; row++)
   //    for (int col = 0; col < N; col++)
   //       {
   //             set_cell(arr1, row, col, rand()%2);
   //       }
   // for (int row = 0; row < N; row++)
   //    for (int col = 0; col < N; col++)
   //       {
   //             set_cell(arr2, row, col, rand()%2);
   //       }
}

uint8_t get_alive(uint8_t * array, uint32_t row, uint32_t col)
{
   uint8_t alive_count = 0;
   for(int i = -1; i <= 1; i++) {
      for(int j = -1; j <= 1; j++){
         uint32_t a_row = (row + i + N) % N;
         uint32_t a_col = (col + j + N) % N;
         if ((get_cell(array, a_row, a_col) == ALIVE)
            && ((i != 0) || (j != 0))){
            alive_count++;
         }
      }
   }
   return alive_count;
}