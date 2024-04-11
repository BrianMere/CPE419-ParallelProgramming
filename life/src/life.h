#ifndef LIFE_H
#define LIFE_H

#include <stdint.h>

#define N 10
#define ALIVE   1
#define DEAD    0

/** Allocate memory for our "frame buffers" */
void init_arrays();



/**
    Get the value of a cell, the parallel way.
*/
inline uint8_t get_cell(uint8_t * array, uint32_t row, uint32_t col)
{
   return array[row * N + col];
}
/**
    Set the value of a cell.
*/
inline void set_cell(uint8_t * array, uint32_t row, uint32_t col, uint8_t value)
{
   array[row * N + col] = value;
}

#endif