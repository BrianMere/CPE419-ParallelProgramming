#ifndef GRAPHICS_H
#define GRAPHICS_H


#include <stdint.h>

/**
    Clear's the screen of all squares.

    Also clears terminal output.
*/
void clear_screen();

/**
    Prints the screen given an array of passed N parameters.

    Assumes that the array has zero (for dead) and non-zero (alive) cells.

    This will clear the screen, then consider the array and console size when printing.
*/
void print_screen(uint8_t * arr, uint32_t n);

/** Gets the number of characters the screen has access to in terms of number of rows. */
int get_rows();

/** Gets the number of characters the screen has access to in number of columns */
int get_columns();

#endif