#include "graphics.h"
#include "life.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>
#elif __APPLE__
    #include <unistd.h>
#elif __linux__
    #include <unistd.h>
#endif

void clear_screen(){
    #if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
        system("clear");
    #endif

    #if defined(_WIN32) || defined(_WIN64)
        system("cls");
    #endif
}

void print_screen(uint8_t * arr, uint32_t n)
{
    clear_screen();

    int width = get_rows();
    int height = get_columns();

    for(uint32_t i = 0; i < n; i++)
    {
        for(uint32_t j = 0; j < n; j++)
        {
            if(get_cell(arr, i, j)) // when alive
                printf("⬜");
            else // when dead
                printf("⬛");
        }
        printf("\n");
    }
}

int get_rows() {
    int rows = 0;

    #ifdef _WIN32 || _WIN64
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
            rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
        }
    #elif __APPLE__ || __linux__ || __unix__
        // Use the stty command to get the number of rows
        FILE *pipe = popen("stty size", "r");
        if (pipe != NULL) {
            char output[20]; // Assuming the output won't be larger than 20 characters
            if (fgets(output, sizeof(output), pipe) != NULL) {
                char *token = strtok(output, " ");
                if (token != NULL) {
                    rows = atoi(token);
                }
            }
            pclose(pipe);
        }
    #endif

    return rows;
}

int get_columns() {
    int columns = 0;

    #ifdef _WIN32 || _WIN64
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
            columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        }
    #elif __APPLE__ || __linux__ || __unix__
        // Use the stty command to get the number of columns
        FILE *pipe = popen("stty size", "r");
        if (pipe != NULL) {
            char output[20]; // Assuming the output won't be larger than 20 characters
            if (fgets(output, sizeof(output), pipe) != NULL) {
                strtok(output, " "); // Discard the first token (rows)
                char *token = strtok(NULL, " ");
                if (token != NULL) {
                    columns = atoi(token);
                }
            }
            pclose(pipe);
        }
    #endif

    return columns;
}
