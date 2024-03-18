/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*******************************************************************************
* @file phy_rx_mimo_mmse.cpp
* @brief MIMO MMSE detection with post SINR.
*******************************************************************************/

/*******************************************************************************
* Include private header files
*******************************************************************************/
#include <map>
#include <tuple>
#include "mimo.hpp"
#include "phy_rx_mimo_mmse.h"
#include "phy_rx_mimo_mmse_internal.h"
#include "phy_tafo_table_gen.h"
#ifdef _BBLIB_AVX512_
using namespace W_SDK;

// Number of symbol groups sharing the same interpolated CE
int16_t numgroups[BBLIB_INTERP_GRANS] = { 14, 6, 4, 3, 2 };

int16_t numgroups_2p2_A[BBLIB_INTERP_GRANS] = { 10, 5, 4, 0, 2 };   // Linear4 is not supported
int16_t numgroups_2p2_B[BBLIB_INTERP_GRANS] = { 10, 0, 4, 3, 2 };   // Linear2 is not supported

// Number of symbols in each group for different granularities
int16_t numingroup[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF] = {
{1,1,1,1,1,1,1,1,1,1,1,1,1,1},
{2,2,2,2,2,2,0,0,0,0,0,0,0,0},
{3,3,3,3,0,0,0,0,0,0,0,0,0,0},
{4,4,4,0,0,0,0,0,0,0,0,0,0,0},
{6,6,0,0,0,0,0,0,0,0,0,0,0,0} };

int16_t numingroup_2p2_A[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF] = {
{1,1,1,1,1,1,1,1,1,1,0,0,0,0},
{2,2,2,2,2,0,0,0,0,0,0,0,0,0},
{2,3,3,2,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{5,5,0,0,0,0,0,0,0,0,0,0,0,0} };

int16_t numingroup_2p2_B[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF] = {
{1,1,1,1,1,1,1,1,1,1,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{2,2,3,3,0,0,0,0,0,0,0,0,0,0},
{4,3,3,0,0,0,0,0,0,0,0,0,0,0},
{5,5,0,0,0,0,0,0,0,0,0,0,0,0} };


// Data symbol numbers belonging to each group Type A
int16_t symnumsA[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6] = {
    { {0},{1},{3},{4},{5},{6},{7},{8},{9},{10},{12},{13},{0},{0} },
    { {0,1},{3,4},{5,6},{7,8},{9,10},{12,13},{0},{0},{0},{0},{0},{0},{0},{0} },
    { {0,1,3},{4,5,6},{7,8,9},{10,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    { {0,1,3,4},{5,6,7,8},{9,10,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    { {0,1,3,4,5,6},{7,8,9,10,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} } };

// Data symbol numbers belonging to each group Type B
int16_t symnumsB[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6] = {
    { {1},{2},{3},{4},{5},{6},{7},{8},{9},{11},{12},{13},{0},{0} },
    { {1,2},{3,4},{5,6},{7,8},{9,11},{12,13},{0},{0},{0},{0},{0},{0},{0},{0} },
    { {1,2,3},{4,5,6},{7,8,9},{11,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    { {1,2,3,4},{5,6,7,8},{9,11,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    { {1,2,3,4,5,6},{7,8,9,11,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} } };

int16_t symnums_2p2_A[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6] = {
    {{0},{1},{4},{5},{6},{7},{8},{9},{12},{13},{0},{0},{0},{0} },
    {{0,1},{4,5},{6,7},{8,9},{12,13},{0,0},{0},{0},{0},{0},{0},{0},{0},{0} },
    {{0,1},{4,5,6},{7,8,9},{12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    {{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    {{0,1,4,5,6},{7,8,9,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} } };

int16_t symnums_2p2_B[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6] = {
    {{2},{3},{4},{5},{6},{7},{8},{11},{12},{13},{0},{0},{0},{0} },
    {{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    {{2,3},{4,5},{6,7,8},{11,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    {{2,3,4,5},{6,7,8},{11,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} },
    {{2,3,4,5,6},{7,8,11,12,13},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} } };

int16_t g_flag_symH_upd[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF] = {
    // A
    {{ 1,1,0,1,1,1,1,1,1,1,1,0,1,1 },
    { 1,0,0,1,0,1,0,1,0,1,0,0,1,0 },
    { 1,0,0,0,1,0,0,1,0,0,1,0,0,0 },
    { 1,0,0,0,0,1,0,0,0,1,0,0,0,0 },
    { 1,0,0,0,0,0,0,1,0,0,0,0,0,0 }},
    // B
    {{ 0,1,1,1,1,1,1,1,1,1,0,1,1,1 },
    { 0,1,0,1,0,1,0,1,0,1,0,0,1,0 },
    { 0,1,0,0,1,0,0,1,0,0,0,1,0,0 },
    { 0,1,0,0,0,1,0,0,0,1,0,0,0,0 },
    { 0,1,0,0,0,0,0,1,0,0,0,0,0,0 }}
    };

int16_t flag_symH_upd_2p2[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF] = {
    // A
    {{ 1,1,0,0,1,1,1,1,1,1,0,0,1,1 },
    { 1,0,0,0,1,0,1,0,1,0,0,0,1,0 },
    { 1,0,0,0,1,0,0,1,0,0,0,0,1,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 1,0,0,0,0,0,0,1,0,0,0,0,0,0 } },
    // B
    {{ 0,0,1,1,1,1,1,1,1,0,0,1,1,1 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,1,0,1,0,1,0,0,0,0,1,0,0 },
    { 0,0,1,0,0,0,1,0,0,0,0,1,0,0 },
    { 0,0,1,0,0,0,0,1,0,0,0,0,0,0 }}
    };
int16_t g_flag_symH_upd_optA[6][BBLIB_INTERP_GRANS] = {
//l0 = 3, pos1 = 7, symbol length = 8, 9
    {0b11101110111111,
    0b10100100101010,
    0b10001000100100,
    0b10000100001000,
    0b10000000100000},
//l0 = 2, pos1 = 7, symbol length = 8, 9
    {0b11011110111111,
    0b10010100101010,
    0b10001000100010,
    0b10000100001000,
    0b10000000100000},
//l0 = 3, pos1 = 9, symbol length = 10, 11, 12
    {0b11101111101111,
    0b10100101001010,
    0b10001001000100,
    0b10000100001000,
    0b10000001000000},
//l0 = 2, pos1 = 9, symbol length = 10, 11, 12
    {0b11011111101111,
    0b10010101001010,
    0b10001001000100,
    0b10000100001000,
    0b10000001000000},
//l0 = 3, pos1 = 11, symbol length = 13, 14
    {0b11101111111011,
    0b10100101010010,
    0b10001001001000,
    0b10000100010000,
    0b10000001000000},
//l0 = 2, pos1 = 11, symbol length = 13, 14
    {0b11011111111011,
    0b10010101010010,
    0b10001001001000,
    0b10000100010000,
    0b10000001000000},
    };
int16_t g_flag_symH_upd_optB[4][BBLIB_INTERP_GRANS] = {
//pos1 = 4 symbol length = 5, 6, 7
    {0b01110111111111,
    0b01010010001000,
    0b01000100100100,
    0b01000010001000,
    0b01000001000000},
//pos1 = 6 symbol length = 8, 9
    {0b01111111110111,
    0b01010101010010,
    0b01001001000100,
    0b01000100010000,
    0b01000001000000},
//pos1 = 8 symbol length = 10, 11
    {0b01111111110111,
    0b01010101010010,
    0b01001001000100,
    0b01000100010000,
    0b01000001000000},
//pos1 = 10 symbol length = 12, 13, 14
    {0b01111111110111,
    0b01010101010010,
    0b01001001000100,
    0b01000100010000,
    0b01000001000000}
    };
int16_t flag_symH_upd_2p2_opt[2][BBLIB_INTERP_GRANS] = {
    // A
    {0b11001111110011,
    0b10001010100010,
    0b10001001000010,
    0b00000000000000,
    0b10000001000000},
    // B
    {0b00111111100111,
    0b00000000000000,
    0b00101010000100,
    0b00100010000100,
    0b00100001000000}
    };
// Weights of first and second DMRS symbols in each group for different granularities Type A
int16_t wType[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2] = {
    {{ {20024, -3640}, {18204, -1820}, {0,0}, {14653, 1820}, {12743, 3640}, {10922, 5461}, {9102, 7281},
      {7281, 9102}, {5461, 10922}, {3640, 12743}, {1820, 14653}, {0,0}, {-1820, 18204}, {-3640, 20024} },

    {{19114, -2730}, {19114, -2730}, {0,0}, {13653, 2730}, {13653, 2730}, {10012, 6371}, {10012, 6371},
     {6371, 10012}, {6371, 10012}, {2730, 13653}, {2730, 13653}, {0,0}, {-2730, 19114}, {-2730, 19114} },

    {{17597, -1213}, {17597, -1213}, {0,0}, {17597, -1213}, {10922, 5461}, {10922, 5461}, {10922, 5461},
     {5461, 10922}, {5461, 10922}, {5461, 10922}, {-1213, 17597}, {0,0}, {-1213, 17597}, {-1213, 17597} },

    {{16384, 0}, {16384, 0}, {0,0}, {16384, 0}, {16384, 0}, {8192, 8192}, {8192, 8192},
     {8192, 8192}, {8192, 8192}, {0, 16384}, {0, 16384}, {0,0}, {0, 16384}, {0, 16384} },

    {{14260, 2123}, {14260, 2123}, {0,0}, {14260, 2123}, {14260, 2123}, {14260, 2123}, {14260, 2123},
     {2123, 14260}, {2123, 14260}, {2123, 14260}, {2123, 14260}, {0,0}, {2123, 14260}, {2123, 14260} } },

// Weights of first and second DMRS symbols in each group for different granularities Type B
    { { {0,0}, {14745, 1638}, {13107, 3276}, {11468, 4915}, {9830, 6553}, {8192, 8192}, {6553, 9830},
      {4915, 11468}, {3276, 13107}, {1638, 14745}, {0,0}, {-1638, 18022}, {-3276, 19660}, {-4915, 21299} },

    { {0,0}, {13926, 2457}, {13926, 2457}, {10649, 5734}, {10649, 5734}, {7372, 9011}, {7372, 9011},
      {4096, 12288}, {4096, 12288}, {0, 16384}, {0,0}, {0, 16384}, {-4096, 20480}, {-4096, 20480} },

    { {0,0}, {13107, 3276}, {13107, 3276}, {13107, 3276}, {8192, 8192}, {8192, 8192}, {8192, 8192},
      {3276, 13107}, {3276, 13107}, {3276, 13107}, {0,0}, {-3276, 19660}, {-3276, 19660}, {-3276, 19660} },

    { {0,0}, {12288, 4096}, {12288, 4096}, {12288, 4096}, {12288, 4096}, {5734, 10649}, {5734, 10649},
      {5734, 10649}, {5734, 10649}, {-2048, 18432}, {0,0}, {-2048, 18432}, {-2048, 18432}, {-2048, 18432} },

    { {0,0}, {10649, 5734}, {10649, 5734}, {10649, 5734}, {10649, 5734}, {10649, 5734}, {10649, 5734},
      {0, 16384}, {0, 16384}, {0, 16384}, {0,0}, {0, 16384}, {0, 16384}, {0, 16384} } }
    };
// Weights of first and second DMRS symbols in each group for different granularities Type A
int16_t wType_A[6][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2] = {
    //l0 = 3, pos1 = 7, symbol length = 8, 9 linear1
    {{ {28762, -12288}, {24576, -8192}, {20480, -4096}, {0, 0}, {12288, 4094}, {8192, 8192}, {4096, 12288},
      {0, 0}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480} },
    //linear2
    {{26624, -10240}, {26624, -10240}, {16384,0}, {0, 0}, {16384, 0}, {6144, 10240}, {6144, 10240},
     {0, 0}, {-6144, 22528}, {-6144, 22528}, {-6144, 22528}, {-6144, 22528}, {-6144, 22528}, {-6144, 22528} },
    //linear3
    {{24576, -8192}, {24576, -8192}, {24576, -8192}, {0, 0}, {8192, 8192}, {8192, 8192}, {8192, 8192},
     {0, 0}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576} },
    //linear4
    {{21504, -5120}, {21504, -5120}, {21504, -5120}, {0, 0}, {21504, -5120}, {0, 16384}, {0, 16384},
     {0, 0}, {0, 16384}, {0, 16384}, {0, 16384}, {0, 16384}, {0, 16384}, {0, 16384} },
    //linear6
    {{16384, 0}, {16384, 0}, {16384, 0}, {0, 0}, {16384, 0}, {16384, 0}, {16384, 0},
     {0, 0}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853} } },
    //l0 = 2, pos1 = 7, symbol length = 8, 9 linear1
     {{ {22938, -6554}, {19661, -3277}, {0,0}, {13107, 3277}, {9830, 6554}, {6554, 9830}, {3277, 13107},
      {0, 0}, {-3277, 19661}, {-3277, 19661}, {-3277, 19661}, {-3277, 19661}, {-3277, 19661}, {-3277, 19661} },
    //linear2
    {{21299, -4915}, {21299, -4915}, {0,0}, {11469, 4915}, {11469, 4915}, {4915, 11469}, {4915, 11469},
     {0, 0}, {-4915, 21299}, {-4915, 21299}, {-4915, 21299}, {-4915, 21299}, {-4915, 21299}, {-4915, 21299} },
    //linear3
    {{18570, -2186}, {18570, -2186}, {0,0}, {18570, -2186}, {6554, 9830}, {6554, 9830}, {6554, 9830},
     {0, 0}, {-6554, 22938}, {-6554, 22938}, {-6554, 22938}, {-6554, 22938}, {-6554, 22938}, {-6554, 22938} },
    //linear4
    {{16384, 0}, {16384, 0}, {0,0}, {16384, 0}, {16384, 0}, {0, 16384}, {0, 16384},
     {0, 0}, {0, 16384}, {0, 16384}, {0, 16384}, {0, 16384}, {0, 16384}, {0, 16384} },
    //linear6
    {{12561, 3823}, {12561, 3823}, {0,0}, {12561, 3823}, {12561, 3823}, {12561, 3823}, {12561, 3823},
     {0, 0}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853}, {-11469, 27853} } },
     //l0 = 3, pos1 = 9, symbol length = 10, 11, 12 linear1
     {{ {24576, -8192}, {21845, -5461}, {19115, -2731}, {0, 0}, {13653, 2731}, {10923, 5461}, {8192, 8192},
      {5461, 10923}, {2731, 13653}, {0, 0}, {-2731, 19115}, {-5461, 21845}, {-5461, 21845}, {-5461, 21845} },
    //linear2
    {{23211, -6827}, {23211, -6827}, {16384,0}, {0, 0}, {16384,0}, {9557, 6827}, {9557, 6827},
     {4096, 12288}, {4096, 12288}, {0, 0}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480} },
    //linear3
    {{21845, -5461}, {21845, -5461}, {21845, -5461}, {0, 0}, {10923, 5461}, {10923, 5461}, {10923, 5461},
     {1820, 14564}, {1820, 14564}, {0, 0}, {1820, 14564}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576} },
    //linear4
    {{19797, -3413}, {19797, -3413}, {19797, -3413}, {0, 0}, {19797, -3413}, {6827, 9557}, {6827, 9557},
     {6827, 9557}, {6827, 9557}, {0, 0}, {-6827, 23211}, {-6827, 23211}, {-6827, 23211}, {-6827, 23211} },
    //linear6
    {{16384, 0}, {16384, 0}, {16384, 0}, {0, 0}, {16384, 0}, {16384, 0}, {16384, 0},
     {-3186, 19570}, {-3186, 19570}, {0, 0}, {-3186, 19570}, {-3186, 19570}, {-3186, 19570}, {-3186, 19570} } },
     //l0 = 2, pos1 = 9, symbol length = 10, 11, 12 linear1
     {{ {21065, -4681}, {18725, -2341}, {0,0}, {14043, 2341}, {11703, 4681}, {9362, 7022}, {7022, 9362},
      {4681, 11703}, {2341, 14043}, {0, 0}, {-2341, 18725}, {-4681, 21065}, {-4681, 21065}, {-4681, 21065} },
    //linear2
    {{19895, -3511}, {19895, -3511}, {0,0}, {12873, 3511}, {12873, 3511}, {8192, 8192}, {8192, 8192},
     {3511, 12873}, {3511, 12873}, {0, 0}, {-3511, 19895}, {-3511, 19895}, {-8192, 24576}, {-8192, 24576} },
    //linear3
    {{17944, -1560}, {17944, -1560}, {0,0}, {17944, -1560}, {9362, 7022}, {9362, 7022}, {9362, 7022},
     {1560, 14824}, {1560, 14824}, {0, 0}, {1560, 14824}, {-7022, 23406}, {-7022, 23406}, {-7022, 23406} },
    //linear4
    {{16384, 0}, {16384, 0}, {0,0}, {16384, 0}, {16384, 0}, {5851, 10533}, {5851, 10533},
     {5851, 10533}, {5851, 10533}, {0, 0}, {-5851, 22235}, {-5851, 22235}, {-5851, 22235}, {-5851, 22235} },
    //linear6
    {{13653, 2731}, {13653, 2731}, {0,0}, {13653, 2731}, {13653, 2731}, {13653, 2731}, {13653, 2731},
     {-2731, 19115}, {-2731, 19115}, {0, 0}, {-2731, 19115}, {-2731, 19115}, {-2731, 19115}, {-2731, 19115} } },
    //l0 = 3, pos1 = 11, symbol length = 13, 14 linear1
     {{ {22528, -6144}, {20480, -4096}, {18432, -2048}, {0, 0}, {14336, 2048}, {12288, 4096}, {10240, 6144},
      {8192, 8192}, {6144, 10240}, {4096, 12288}, {2048, 14336}, {0,0}, {-2048, 18432}, {-4096, 20480} },
    //linear2
    {{21504, -5120}, {21504, -5120}, {16384, 0}, {0, 0}, {16384, 0}, {11264, 5120}, {11264, 5120},
     {7168, 9216}, {7168, 9216}, {3072, 13312}, {3072, 13312}, {0,0}, {-3072, 19456}, {-3072, 19456} },
    //linear3
    {{20480, -4096}, {20480, -4096}, {20480, -4096}, {0, 0}, {12288, 4096}, {12288, 4096}, {12288, 4096},
     {6144, 10240}, {6144, 10240}, {6144, 10240}, {-1365, 17749}, {0,0}, {-1365, 17749}, {-1365, 17749} },
    //linear4
    {{18944, -2560}, {18944, -2560}, {18944, -2560}, {0, 0}, {18944, -2560}, {9216, 7168}, {9216, 7168},
     {9216, 7168}, {9216, 7168}, {0, 16384}, {0, 16384}, {0,0}, {0, 16384}, {0, 16384} },
    //linear6
    {{16384, 0}, {16384, 0}, {16384, 0}, {0, 0}, {16384, 0}, {16384, 0}, {16384, 0},
     {2389, 13995}, {2389, 13995}, {2389, 13995}, {2389, 13995}, {0,0}, {2389, 13995}, {2389, 13995} } },
    //l0 = 2, pos1 = 11, symbol length = 13, 14 linear1
     {{ {20024, -3640}, {18204, -1820}, {0,0}, {14653, 1820}, {12743, 3640}, {10922, 5461}, {9102, 7281},
      {7281, 9102}, {5461, 10922}, {3640, 12743}, {1820, 14653}, {0,0}, {-1820, 18204}, {-3640, 20024} },
    //linear2
    {{19114, -2730}, {19114, -2730}, {0,0}, {13653, 2730}, {13653, 2730}, {10012, 6371}, {10012, 6371},
     {6371, 10012}, {6371, 10012}, {2730, 13653}, {2730, 13653}, {0,0}, {-2730, 19114}, {-2730, 19114} },
    //linear3
    {{17597, -1213}, {17597, -1213}, {0,0}, {17597, -1213}, {10922, 5461}, {10922, 5461}, {10922, 5461},
     {5461, 10922}, {5461, 10922}, {5461, 10922}, {-1213, 17597}, {0,0}, {-1213, 17597}, {-1213, 17597} },
    //linear4
    {{16384, 0}, {16384, 0}, {0,0}, {16384, 0}, {16384, 0}, {8192, 8192}, {8192, 8192},
     {8192, 8192}, {8192, 8192}, {0, 16384}, {0, 16384}, {0,0}, {0, 16384}, {0, 16384} },
    //linear6
    {{14260, 2123}, {14260, 2123}, {0,0}, {14260, 2123}, {14260, 2123}, {14260, 2123}, {14260, 2123},
     {2123, 14260}, {2123, 14260}, {2123, 14260}, {2123, 14260}, {0,0}, {2123, 14260}, {2123, 14260} } } };

// Weights of first and second DMRS symbols in each group for different granularities Type B
int16_t wType_B[4][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2] = {
    //pos1 = 4 symbol length = 5, 6, 7 linear1
    {{ {0,0}, {12288, 4096}, {8192, 8192}, {4096, 12288}, {0, 0}, {-4096, 20480}, {-8192, 24576},
      {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576} },
    //linear2
    { {0,0}, {10240, 6144}, {10240, 6144}, {0, 16384}, {0, 0}, {0, 16384}, {-10240, 26624},
      {-10240, 26624}, {-10240, 26624}, {-10240, 26624}, {-10240, 26624}, {-10240, 26624}, {-10240, 26624}, {-10240, 26624} },
    //linear3
    { {0,0}, {8192, 8192}, {8192, 8192}, {8192, 8192}, {0, 0}, {-8192, 24576}, {-8192, 24576},
      {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576} },
    //linear4
    { {0,0}, {5120, 11264}, {5120, 11264}, {5120, 11264}, {0, 0}, {5120, 11264}, {-14336, 30720},
      {-14336, 30720}, {-14336, 30720}, {-14336, 30720}, {-14336, 30720}, {-14336, 30720}, {-14336, 30720}, {-14336, 30720} },
    //linear6
    { {0,0}, {0, 16384}, {0, 16384}, {0, 16384}, {0, 0}, {0, 16384}, {0, 16384},
      {0, 16384}, {0, 16384}, {0, 16384}, {0,0}, {0, 16384}, {0, 16384}, {0, 16384} } },
    //pos1 = 6 symbol length = 8, 9 linear1
    {{ {0,0}, {13653, 2731}, {10923, 5461}, {8192, 8192}, {5461, 5461}, {2731, 13653}, {0, 0},
      {-2731, 19115}, {-5461, 21845}, {-5461, 21845}, {-5461, 21845}, {-5461, 21845}, {-5461, 21845}, {-5461, 21845} },
    //linear2
    { {0,0}, {12288, 4096}, {12288, 4096}, {6827, 9557}, {6827, 9557}, {0, 16384}, {0, 0},
      {0, 16384}, {-6827, 23211}, {-6827, 23211}, {-6827, 23211}, {-6827, 23211}, {-6827, 23211}, {-6827, 23211} },
    //linear3
    { {0,0}, {10923, 5461}, {10923, 5461}, {10923, 5461}, {1820, 14564}, {1820, 14564}, {0, 0},
      {1820, 14564}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576}, {-8192, 24576} },
    //linear4
    { {0,0}, {9557, 6827}, {9557, 6827}, {9557, 6827}, {9557, 6827}, {-3413, 19797}, {0, 0},
      {-3413, 19797}, {-3413, 19797}, {-3413, 19797}, {-3413, 19797}, {-3413, 19797}, {-3413, 19797}, {-3413, 19797} },
    //linear6
    { {0,0}, {6372, 10012}, {6372, 10012}, {6372, 10012}, {6372, 10012}, {6372, 10012}, {0, 0},
      {6372, 10012}, {-12288, 28672}, {-12288, 28672}, {-12288, 28672}, {-12288, 28672}, {-12288, 28672}, {-12288, 28672} } },
    //pos1 = 8 symbol length = 10, 11 linear1
    {{ {0,0}, {14336, 2048}, {12288, 4096}, {10240, 6144}, {8192, 8192}, {6144, 10240}, {4096, 12288},
      {2048, 14336}, {0, 0}, {-2048, 18432}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480}, {-4096, 20480} },
    //linear2
    { {0,0}, {13312, 3072}, {13312, 3072}, {9216, 7168}, {9216, 7168}, {5120, 11264}, {5120, 11264},
      {0, 16384}, {0, 0}, {0, 16384}, {-5120, 21504}, {-5120, 21504}, {-5120, 21504}, {-5120, 21504} },
    //linear3
    { {0,0}, {12288, 4096}, {12288, 4096}, {12288, 4096}, {6144, 10240}, {6144, 10240}, {6144, 10240},
      {-1365, 17749}, {0, 0}, {-1365, 17749}, {-1365, 17749}, {-1365, 17749}, {-1365, 17749}, {-1365, 17749} },
    //linear4
    { {0,0}, {11264, 5120}, {11264, 5120}, {11264, 5120}, {11264, 5120}, {2560, 13824}, {2560, 13824},
      {2560, 13824}, {0, 0}, {2560, 13824}, {-7168, 23552}, {-7168, 23552}, {-7168, 23552}, {-7168, 23552} },
    //linear6
    { {0,0}, {9216, 7168}, {9216, 7168}, {9216, 7168}, {9216, 7168}, {9216, 7168}, {9216, 7168},
      {-4779, 21163}, {0, 0}, {-4779, 21163}, {-4779, 21163}, {-4779, 21163}, {-4779, 21163}, {-4779, 21163} }},
    //pos1 = 10 symbol length = 12, 13, 14 linear1
    {{ {0,0}, {14745, 1638}, {13107, 3276}, {11468, 4915}, {9830, 6553}, {8192, 8192}, {6553, 9830},
      {4915, 11468}, {3276, 13107}, {1638, 14745}, {0,0}, {-1638, 18022}, {-3276, 19660}, {-4915, 21299} },
    //linear2
    { {0,0}, {13926, 2457}, {13926, 2457}, {10649, 5734}, {10649, 5734}, {7372, 9011}, {7372, 9011},
      {4096, 12288}, {4096, 12288}, {0, 16384}, {0,0}, {0, 16384}, {-4096, 20480}, {-4096, 20480} },
    //linear3
    { {0,0}, {13107, 3276}, {13107, 3276}, {13107, 3276}, {8192, 8192}, {8192, 8192}, {8192, 8192},
      {3276, 13107}, {3276, 13107}, {3276, 13107}, {0,0}, {-3276, 19660}, {-3276, 19660}, {-3276, 19660} },
    //linear4
    { {0,0}, {12288, 4096}, {12288, 4096}, {12288, 4096}, {12288, 4096}, {5734, 10649}, {5734, 10649},
      {5734, 10649}, {5734, 10649}, {-2048, 18432}, {0,0}, {-2048, 18432}, {-2048, 18432}, {-2048, 18432} },
    //linear6
    { {0,0}, {10649, 5734}, {10649, 5734}, {10649, 5734}, {10649, 5734}, {10649, 5734}, {10649, 5734},
      {0, 16384}, {0, 16384}, {0, 16384}, {0,0}, {0, 16384}, {0, 16384}, {0, 16384} } }};

// Weights of first and second half slot DMRS symbols in each group for different granularities for 2+2 DMRS Type A
int16_t wType_2p2[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2] = {
    {{ {21504, -5120}, {19456, -3072}, {0,0}, {0, 0}, {13312, 3072}, {11264, 5120}, {9216, 7168},
      {7168, 9216}, {5120, 11264}, {3072, 13312}, {0, 0}, {0,0}, {-3072, 19456}, {-5120, 21504} },

    {{20480, -4096}, {20480, -4096}, {0,0}, {0, 0}, {12288, 4096}, {12288, 4096}, {8192, 8192},
     {8192, 8192}, {4096, 12288}, {4096, 12288}, {0, 0}, {0,0}, {-4096, 20480}, {-4096, 20480} },

    {{20480, -4096}, {20480, -4096}, {0,0}, {0, 0}, {11264, 5120}, {11264, 5120}, {11264, 5120},
     {5120, 11264}, {5120, 11264}, {5120, 11264}, {0, 0}, {0,0}, {-4096, 20480}, {-4096, 20480} },

    {{16384, 0}, {16384, 0}, {0,0}, {0, 0}, {16384, 0}, {8192, 8192}, {8192, 8192},
     {8192, 8192}, {8192, 8192}, {0, 16384}, {0, 0}, {0,0}, {0, 16384}, {0, 16384} },

    {{14950, 1433}, {14950, 1433}, {0,0}, {0, 0}, {14950, 1433}, {14950, 1433}, {14950, 1433},
     {1433, 14950}, {1433, 14950}, {1433, 14950}, {0, 0}, {0,0}, {1433, 14950}, {1433, 14950} } },

// Weights of first and second half slot DMRS symbols in each group for different granularities for 2+2 DMRS Type B
    { { {0,0}, {0, 0}, {13653, 2730}, {11832, 4551}, {10112, 6371}, {8192, 8192}, {6371, 10112},
      {4551, 11832}, {2730, 13653}, {0, 0}, {0,0}, {-2730, 19114}, {-4551, 20935}, {-6371, 22755} },

    { {0,0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
      {0, 0}, {0, 0}, {0, 0}, {0,0}, {0, 0}, {0, 0}, {0, 0} },

    { {0,0}, {0, 0}, {12743, 3640}, {12743, 3640}, {9102, 7281}, {9102, 7281}, {4551, 11832},
      {4551, 11832}, {4551, 11832}, {0, 0}, {0,0}, {-4551, 20935}, {-4551, 20935}, {-4551, 20935} },

    { {0,0}, {0, 0}, {10922, 5461}, {10922, 5461}, {10922, 5461}, {10922, 5461}, {4551, 11832},
      {4551, 11832}, {4551, 11832}, {0, 0}, {0,0}, {-4551, 20935}, {-4551, 20935}, {-4551, 20935} },

    { {0,0}, {0, 0}, {10012, 6371}, {10012, 6371}, {10012, 6371}, {10012, 6371}, {10012, 6371},
      {-1274, 17658}, {-1274, 17658}, {0, 0}, {0,0}, {-1274, 17658}, {-1274, 17658}, {-1274, 17658} } }
    };

//This table is to compensate for effect of long CP effect on FOC
float FocPhaseFixTable[2][BBLIB_N_SYMB_PER_SF] = {
    {0, 1, 2, 3, 4, 5, 6, 7.007299, 8.007299, 9.007299, 10.007299, 11.007299, 12.007299, 13.007299}, //for SCS=15kHz
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13} //for other SCS
};

static const auto m512shuffleIQ = I8vec64(
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);

static const auto m512switchIQ = I8vec64(
    61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50,
    45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34,
    29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18,
    13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

static const auto m512NegI = I16vec32(
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF);


//    15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
/*! \brief MMSE MIMO detection for 16TX16R, with post SINR calculation.
    \param [in] request Input request structure for MMSE MIMO.
    \param [out] response Output response structure for MMSE MIMO..
*/

template<int N = 16>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[N][N], F32vec16 matBIm[N][N],
    F32vec16 matInvBRe[N][N], F32vec16 matInvBIm[N][N]);

template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[16][16], F32vec16 matBIm[16][16],
    F32vec16 matInvBRe[16][16], F32vec16 matInvBIm[16][16]) {
    #define type_cast reinterpret_cast<__m512 (*)[16]>
    matrix_inv_cholesky_16x16(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[8][8], F32vec16 matBIm[8][8],
    F32vec16 matInvBRe[8][8], F32vec16 matInvBIm[8][8]) {
    #define type_cast reinterpret_cast<__m512 (*)[8]>
    matrix_inv_cholesky_8x8(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[7][7], F32vec16 matBIm[7][7],
    F32vec16 matInvBRe[7][7], F32vec16 matInvBIm[7][7]) {
    #define type_cast reinterpret_cast<__m512 (*)[7]>
    matrix_inv_cholesky_7x7(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[6][6], F32vec16 matBIm[6][6],
    F32vec16 matInvBRe[6][6], F32vec16 matInvBIm[6][6]) {
    #define type_cast reinterpret_cast<__m512 (*)[6]>
    matrix_inv_cholesky_6x6(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[5][5], F32vec16 matBIm[5][5],
    F32vec16 matInvBRe[5][5], F32vec16 matInvBIm[5][5]) {
    #define type_cast reinterpret_cast<__m512 (*)[5]>
    matrix_inv_cholesky_5x5(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[4][4], F32vec16 matBIm[4][4],
    F32vec16 matInvBRe[4][4], F32vec16 matInvBIm[4][4]) {
    #define type_cast reinterpret_cast<__m512 (*)[4]>
    // matrix_inv_cholesky_4x4(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    // matrix_inv_lemma_4x4(ftempBRe, ftempBIm, finvARe, finvAIm, BBLIB_MMSE_LEMMA_SCALING);
    matrix_inv_lemma_4x4(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm),
        BBLIB_MMSE_LEMMA_SCALING);
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[3][3], F32vec16 matBIm[3][3],
    F32vec16 matInvBRe[3][3], F32vec16 matInvBIm[3][3]) {
    #define type_cast reinterpret_cast<__m512 (*)[3]>
    matrix_inv_cholesky_3x3(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[2][2], F32vec16 matBIm[2][2],
    F32vec16 matInvBRe[2][2], F32vec16 matInvBIm[2][2]) {
    #define type_cast reinterpret_cast<__m512 (*)[2]>
    matrix_inv_cholesky_2x2(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[1][1], F32vec16 matBIm[1][1],
    F32vec16 matInvBRe[1][1], F32vec16 matInvBIm[1][1]) {
        matInvBRe[0][0] = rcp(matBRe[0][0]);
        matInvBIm[0][0] = matBIm[0][0];
}

#define IM_NEG_RE(a, b)  b = shuffle(static_cast<I8vec64>(a), \
                        static_cast<I8vec64>(m512switchIQ));\
                        b = masksub(b, 0x55555555, T(), b);
// mmse mimo linear interpolation
template<size_t N_RX = 16, size_t N_TX = 16> struct MimoMmse {
    // mmse mimo
#ifdef _BBLIB_SPR_
    static void mimo_mmse_avx512(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response, CF16vec16 d) {
        using T = CF16vec16;
        using FloatType = typename DataType<T>::FloatType;
        using Float = typename DataType<T>::Float;
        T *pChIn[N_TX][N_RX];
        T ChIn[N_TX][N_RX];
        T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX];
        T *pTx[N_RX][BBLIB_N_SYMB_PER_SF];
        T offsetExp[N_TX];

        FloatType ftempARe[N_TX][N_TX];//float A = H'*H + sigam2
        FloatType ftempZRe[N_TX];//float Z=H'*Y
        FloatType ftempBRe[N_TX][N_TX];

        const auto mmse_x_left = (Float)(1 << BBLIB_MMSE_X_LEFT_SHIFT);
        //sigma
        T rxIn[N_RX];
        const Float sigma2 = (Float)(request->nSigma2);// for function debug
        const auto left_shift = mmse_x_left;
        const auto avxShift = FloatType(left_shift, 0.0);
        const auto avxfSigma2 = FloatType(sigma2, 0.0);
        float *pPostSinr[N_TX];

        int16_t nFftSize = request->nFftSize;
        int16_t nMaxCp = 0, nMinCp = 0;
        int16_t *pFoCompScCp = request->pFoCompScCp;

        int16_t nNumerology = request->nNumerology;
        if (request->nEnableFoComp)
        {
            if (unlikely(-1 == bblib_get_sys_params(nNumerology, nFftSize, &nMaxCp, &nMinCp, NULL)))
            {
                printf("Error! Not support this case nMu %d nFftSize %d\n", nNumerology, nFftSize);
                return;
            }
        }
        const int16_t nSubCarrier = request->nSubCarrier;
        const int16_t nAlignedTotalSubCarrier = request->nTotalAlignedSubCarrier;
        const int16_t nStartSC = request->nStartSC;
        const int16_t nTime = nSubCarrier / 16;
        const int16_t nRestLen = nSubCarrier - nTime * 16;
        const int16_t nChSymb = request->nChSymb;
        int16_t nStartSymbIndex = 0;

        // if (request->nEnableFoComp) {
        //     auto a = foParaMap[nNumerology][nFftSize];
        //     pFoCompScCp = std::get<0>(a);
        //     nMaxCp = std::get<1>(a);
        //     nMinCp = std::get<2>(a);
        // }
        // loop channel symbol
        for (size_t iChSymb = 0; iChSymb < nChSymb; iChSymb++) {
            int16_t nSymNum = request->pSymbIndex[iChSymb];

            for (size_t j = 0; j < N_TX; j++) {

                if (request->nEnableFoComp) {
                    auto decompOffset = nSymNum * (nFftSize + nMinCp) + nMaxCp;
                    decompOffset = floor(decompOffset * request->fEstCfo[j] * nFftSize);
                    decompOffset = (decompOffset + nFftSize) % nFftSize;
                    // auto pTempX = reinterpret_cast<T *>(&pFoCompScCp[decompOffset * 2 * 16]);
                    // offsetExp[j] = loadu(pTempX);
                    auto pTempX = reinterpret_cast<int32_t *>(&pFoCompScCp[decompOffset * 2]);
                    offsetExp[j] = _mm512_set1_epi32(*pTempX);
                }

                // convert channel coefficient pointer
                for (size_t i = 0; i < N_RX; i ++) {
                    pChIn[j][i] = reinterpret_cast<T *>(
                        reinterpret_cast<int32_t *>(request->pChState[i][j]) +
                        iChSymb * nAlignedTotalSubCarrier + nStartSC);

                    // if (request->nEnableFoComp) {
                    //     // Decompensate FO for channel
                    //     auto ch = loadu(pChIn[j][i]);
                    //     *pChIn[j][i] = fmul(ch, offsetExp[j]);
                    // }
                    // convert rx pointer
                    for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                        const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                        pRxIn[nDataSymbIdx][i] = reinterpret_cast<T *>
                            (reinterpret_cast<int32_t *>(request->pRxSignal[i][nDataSymbIdx]) + nStartSC);
                    }
                }
            }
            // #pragma unroll(N_TX)
            for (size_t i = 0; i < N_TX; i ++) {
                pPostSinr[i] = reinterpret_cast<float *>(response->pPostSINR[i])
                                + iChSymb * nAlignedTotalSubCarrier + nStartSC;
                for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                    const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                    pTx[i][nDataSymbIdx] = reinterpret_cast<T *>(
                        reinterpret_cast<int32_t *>(response->pEstTxSignal[i][nDataSymbIdx]) + nStartSC);
                }
            }
            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {
                // Mask16 kMask = (nSCIdx + 16 > nSubCarrier) ?
                //         (static_cast<Mask16>(1) << nRestLen) - 1 : 0xffff;
                Mask16 scFlag = (nSCIdx + 16 > nSubCarrier);
                Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
                //calculate the real part of H' * H
                // #pragma unroll(N_TX)
                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        ChIn[j][i] = loadu(pChIn[j][i]);
                        pChIn[j][i]++;
                        _mm_prefetch(pChIn[j][i], _MM_HINT_T2);
                    }
                }

                //1. A = H' * H + Sigma2
                HxH<T, N_TX, N_RX> ( ftempARe, ftempBRe, ChIn, avxfSigma2);

                // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                matrix_inverse<T, N_TX>(ftempBRe);

                #pragma unroll(N_TX)
                for (size_t i = 0; i < N_TX; i++) {
                    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
                    auto gain = postSINRCalc (i, ftempBRe, ftempARe);
                    // temp used
                    auto temp = duplicateReal(gain);
                    storeu(reinterpret_cast<T *>(pPostSinr[i] + nSCIdx), kMask, temp);
                }

                for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                    const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);

                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i ++) {
                        rxIn[i] = loadu(pRxIn[nDataSymbIdx][i]);
                        pRxIn[nDataSymbIdx][i] ++;
                        _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T2);
                    }
                    // 3. Z = H' * y
                    #pragma unroll(N_TX)
                    for (size_t i = 0; i < N_TX; i++) {
                        // calculate the real part of z
                        // calculate the imag part of z
                        // ftempZRe[i] = acc_sum<T, N_RX>(ChIn[i], pRxIn[nDataSymbIdx]);
                        ftempZRe[i] = dotC<T, N_RX>(rxIn, ChIn[i]);
                    }

                    // 4. x = invA * z
                    #pragma unroll(N_TX)
                    for(size_t i = 0; i < N_TX; i ++) {
                        // real imag part
                        auto tx = dot<T, N_TX>(ftempBRe[i], ftempZRe);
                        tx = tx * avxShift;
                        // store x
                        storeu(pTx[i][nDataSymbIdx], kMask, tx);
                        pTx[i][nDataSymbIdx] ++;
                        _mm_prefetch(pTx[i][nDataSymbIdx] , _MM_HINT_T2);
                    }
                }
            }//end the Symbol cycle
            nStartSymbIndex += request->nSymbPerDmrs[iChSymb];
        }
    }
#endif

    static void mimo_mmse_avx512(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response, Is16vec32 d) {
        using T = Is16vec32;
        using FloatType = typename DataType<T>::FloatType;
        using Float = typename DataType<T>::Float;
        T *pChIn[N_TX][N_RX];
        T ChIn[N_TX][N_RX];
        T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX];
        T *pTx[N_RX][BBLIB_N_SYMB_PER_SF];
        T offsetExp[N_TX];

        FloatType ftempARe[N_TX][N_TX];//float A = H'*H + sigam2
        FloatType ftempZRe[N_TX];//float Z=H'*Y
        FloatType ftempBRe[N_TX][N_TX];

        T ChImNegRe[N_TX][N_RX];
        FloatType ftempAIm[N_TX][N_TX];//float A = H'*H + sigam2
        FloatType finvAIm[N_TX][N_TX];//float invA
        FloatType finvARe[N_TX][N_TX];//float invA
        FloatType ftempZIm[N_TX];//float Z=H'*Y
        FloatType ftempBIm[N_TX][N_TX];
        const auto mmse_x_left = (Float)(1 << BBLIB_MMSE_X_LEFT_SHIFT);

        const Float sigma2 = (Float)(request->nSigma2);
        const auto left_shift = (N_TX == 4 ? 1.0 / mmse_x_left : mmse_x_left);
        const auto avxShift = FloatType(left_shift);
        const auto avxfSigma2 = FloatType(sigma2);
        const auto nFactor = 1.0 / (Float)(1 << BBLIB_MMSE_LEMMA_SCALING);
        const auto avxGainShift = (N_TX == 4 ? FloatType(nFactor) : FloatType(1.0));
        float *pPostSinr[N_TX];

        int16_t nFftSize = request->nFftSize;
        int16_t nMaxCp = 0, nMinCp = 0;
        int16_t *pFoCompScCp = request->pFoCompScCp;;

        int16_t nNumerology = request->nNumerology;
        if (request->nEnableFoComp)
        {
            if (unlikely(-1 == bblib_get_sys_params(nNumerology, nFftSize, &nMaxCp, &nMinCp, NULL)))
            {
                printf("Error! Not support this case nMu %d nFftSize %d\n", nNumerology, nFftSize);
                return;
            }
        }
        const int16_t nSubCarrier = request->nSubCarrier;
        const int16_t nAlignedTotalSubCarrier = request->nTotalAlignedSubCarrier;
        const int16_t nStartSC = request->nStartSC;
        const int16_t nTime = nSubCarrier / 16;
        const int16_t nRestLen = nSubCarrier - nTime * 16;
        const int16_t nChSymb = request->nChSymb;
        int16_t nStartSymbIndex = 0;

        // if (request->nEnableFoComp) {
        //     auto a = foParaMap[nNumerology][nFftSize];
        //     pFoCompScCp = std::get<0>(a);
        //     nMaxCp = std::get<1>(a);
        //     nMinCp = std::get<2>(a);
        // }
        // loop channel symbol

        for (size_t iChSymb = 0; iChSymb < nChSymb; iChSymb++) {
            int16_t nSymNum = request->pSymbIndex[iChSymb];

            for (size_t j = 0; j < N_TX; j++) {

                if (request->nEnableFoComp) {
                    auto decompOffset = nSymNum * (nFftSize + nMinCp) + nMaxCp;
                    decompOffset = floor(decompOffset * request->fEstCfo[j] * nFftSize);
                    decompOffset = (decompOffset + nFftSize) % nFftSize;
                    // auto pTempX = reinterpret_cast<T *>(&pFoCompScCp[decompOffset * 2 * 16]);
                    // offsetExp[j] = loadu(pTempX);
                    auto pTempX = reinterpret_cast<int32_t *>(&pFoCompScCp[decompOffset * 2]);
                    offsetExp[j] = _mm512_set1_epi32(*pTempX);
                }

                // convert channel coefficient pointer
                for (size_t i = 0; i < N_RX; i ++) {
                    pChIn[j][i] = reinterpret_cast<T *>(
                        reinterpret_cast<int32_t *>(request->pChState[i][j]) +
                        iChSymb * nAlignedTotalSubCarrier + nStartSC);

                    // if (request->nEnableFoComp) {
                    //     // Decompensate FO for channel
                    //     auto ch = loadu(pChIn[j][i]);
                    //     *pChIn[j][i] = fmul(ch, offsetExp[j]);
                    // }
                    // convert rx pointer
                    for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                        const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                        pRxIn[nDataSymbIdx][i] = reinterpret_cast<T *>
                            (reinterpret_cast<int32_t *>(request->pRxSignal[i][nDataSymbIdx]) + nStartSC);
                    }
                }
            }
            // #pragma unroll(N_TX)
            for (size_t i = 0; i < N_TX; i ++) {
                pPostSinr[i] = reinterpret_cast<float *>(response->pPostSINR[i])
                                + iChSymb * nAlignedTotalSubCarrier + nStartSC;
                for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                    const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                    pTx[i][nDataSymbIdx] = reinterpret_cast<T *>(
                        reinterpret_cast<int32_t *>(response->pEstTxSignal[i][nDataSymbIdx]) + nStartSC);
                }
            }
            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {
                // Mask16 kMask = (nSCIdx + 16 > nSubCarrier) ?
                //         (static_cast<Mask16>(1) << nRestLen) - 1 : 0xffff;
                Mask16 scFlag = (nSCIdx + 16 > nSubCarrier);
                Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        ChIn[j][i] = loadu(pChIn[j][i]);
                        pChIn[j][i]++;
                        _mm_prefetch(pChIn[j][i], _MM_HINT_T2);
                        IM_NEG_RE(ChIn[j][i], ChImNegRe[j][i]);
                    }
                }
                //1. A = H' * H + Sigma2
                //calculate the real part of H' * H
                // #pragma unroll(N_TX)
                for (size_t i = 0; i < N_TX; i ++) {
                    ftempARe[i][i] = acc_sum<N_RX>(ChIn[i], ChIn[i]);
                    ftempAIm[i][i] = FloatType(0.0);
                    // ftempAIm[i][i] = allfZero;
                    // B = A + sigma2
                    ftempBRe[i][i] = ftempARe[i][i] + avxfSigma2;
                    ftempBIm[i][i] = FloatType(0.0);
                }

                HxHReal<N_TX, N_RX> ( ftempARe, ftempBRe, ChIn);

                // calculate the imag part of H' * H
                HxHImage<N_TX, N_RX> (ftempAIm, ftempBIm, ChImNegRe, ChIn);

                // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                matrix_inverse<N_TX>(ftempBRe, ftempBIm, finvARe, finvAIm);

                #pragma unroll(N_TX)
                for (size_t i = 0; i < N_TX; i++) {
                    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
                    auto temp = postSINRCalc (i, avxGainShift, finvAIm, finvARe, ftempAIm, ftempARe);
                    storeu(pPostSinr[i] + nSCIdx, kMask, temp);
                }

                for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                    const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                    // 3. Z = H' * y
                    #pragma unroll(N_TX)
                    for (size_t i = 0; i < N_TX; i++) {
                        // calculate the real part of z
                        ftempZRe[i] = acc_sum<N_RX>(ChIn[i], pRxIn[nDataSymbIdx]);
                        // calculate the imag part of z
                        ftempZIm[i] = acc_sum<N_RX>(ChImNegRe[i], pRxIn[nDataSymbIdx]);
                    }

                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i ++) {
                        pRxIn[nDataSymbIdx][i] ++;
                        _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T2);
                    }
                    // 4. x = invA * z
                    #pragma unroll(N_TX)
                    for(size_t i = 0; i < N_TX; i ++) {
                        auto tx = txCalc<N_TX>(i, avxShift, finvAIm, finvARe, ftempZRe, ftempZIm);
                        // store x
                        storeu(pTx[i][nDataSymbIdx], kMask, tx);
                        pTx[i][nDataSymbIdx] ++;
                    }
                }
            }//end the Symbol cycle
            nStartSymbIndex += request->nSymbPerDmrs[iChSymb];
        }
    }

#ifdef _BBLIB_SPR_
    static void mimo_mmse_avx512_lin_interp(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response, CF16vec16 d) {
        using T = CF16vec16;
        using FloatType = typename DataType<T>::FloatType;
        using Float = typename DataType<T>::Float;
        T ChIn[N_TX][N_RX];
        T * pRxIn[N_RX];
        T offsetExp[BBLIB_N_SYMB_PER_SF][N_TX];
        Float wSymAvx[BBLIB_N_SYMB_PER_SF][2];
        T rxIn[N_RX];
        FloatType ftempARe[N_TX][N_TX];//float A = H'*H + sigam2
        FloatType ftempZRe[N_TX];//float Z=H'*Y

        FloatType ftempBRe[N_TX][N_TX];
        #ifdef CVT_ST_LD
        F32vec16 ftempPostSINR[N_TX];
        #else
        FloatType ftempPostSINR[N_TX];
        #endif

        const auto mmse_x_left = (Float)(1 << BBLIB_MMSE_X_LEFT_SHIFT);
        //sigma
        const Float sigma2 = (Float)(request->nSigma2);// for function debug
        const auto left_shift = mmse_x_left;
        const auto avxShift = FloatType(left_shift, 0.0);
        const auto avxfSigma2 = FloatType(sigma2, 0.0);

        int16_t refSym = 0;
        int32_t compOffset;

        int16_t nFftSize = request->nFftSize;
        int16_t nMaxCp = 0, nMinCp = 0;
        int16_t nNumerology = request->nNumerology;
        int16_t *pFoCompScCp = request->pFoCompScCp;
        if (request->nEnableFoComp)
        {
            if (unlikely(-1 == bblib_get_sys_params(nNumerology, nFftSize, &nMaxCp, &nMinCp, NULL)))
            {
                printf("Error! Not support this case nMu %d nFftSize %d\n", nNumerology, nFftSize);
                return;
            }
        }
        const int16_t nSubCarrier = request->nSubCarrier;
        const int16_t nAlignedTotalSubCarrier = request->nTotalAlignedSubCarrier;
        const int16_t nStartSC = request->nStartSC;
        const int16_t nSymb = request->nSymb;
        const int16_t nTime = nSubCarrier / 16;
        const int16_t nRestLen = nSubCarrier - nTime * 16;

        int16_t flag_symH_upd[BBLIB_N_SYMB_PER_SF] = { 0 };
        const size_t offsetDMRS =  N_RX * N_TX * nAlignedTotalSubCarrier;
        int16_t nDmrsChSymb = request->nDmrsChSymb;
        if (nDmrsChSymb != 2) {
            printf("Wrong API: only two DMRS symbols are supported currently with linear interpolation\n");
            return;
        } else {
            int16_t nMappingType = request->nMappingType;
            int16_t nDataSymb = request->nSymb;
            int16_t nGranularity = request->nGranularity;
    #define WCOEFF 16384.0
            for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                auto nSymNum = request->pSymbIndex[iSymb];
                auto wSym0 = wType[nMappingType][nGranularity][nSymNum][0];
                auto wSym1 = wType[nMappingType][nGranularity][nSymNum][1];
                wSymAvx[nSymNum][0] = (static_cast<float>(wSym0) / WCOEFF);
                wSymAvx[nSymNum][1] = (static_cast<float>(wSym1) / WCOEFF);
            }
            for (size_t i = 0; i < BBLIB_N_SYMB_PER_SF; i++) {
                flag_symH_upd[i] = g_flag_symH_upd[nMappingType][nGranularity][i];
            }
    #undef WCOEFF

            if (request->nEnableFoComp)
            {
                // auto a = foParaMap[nNumerology][nFftSize];
                // pFoCompScCp = std::get<0>(a);
                // nMaxCp = std::get<1>(a);
                // nMinCp = std::get<2>(a);

                // Compute FO compensation for each symbol
                for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++)
                {
                    auto nSymNum = request->pSymbIndex[iSymb];
                    if (flag_symH_upd[nSymNum] == 1) {
                        refSym = nSymNum;
                        compOffset = nSymNum * (nFftSize + nMinCp) + nMaxCp;
                    } else {
                        compOffset = (refSym - nSymNum)*(nFftSize + nMinCp);
                    }
                    for (size_t j = 0; j < N_TX; j++) {
                        int16_t compOffsetTemp = floor(compOffset * request->fEstCfo[j] * nFftSize);
                        compOffsetTemp = (compOffsetTemp + nFftSize) % nFftSize;
                        // auto pTempX = reinterpret_cast<T *>(&pFoCompScCp[compOffsetTemp * 2 * 16]);
                        // offsetExp[nSymNum][j] = loadu(pTempX);
                        auto pTempX = reinterpret_cast<int32_t *>(&pFoCompScCp[compOffsetTemp * 2]);
                        offsetExp[nSymNum][j] = _mm512_set1_epi32(*pTempX);
                    }
                }
            }

            // loop channel symbol
            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {
                // Mask16 kMask = (nSCIdx + 16 > nSubCarrier) ?
                //         (static_cast<Mask16>(1) << nRestLen) - 1 : 0xffff;
                Mask16 scFlag = (nSCIdx + 16 > nSubCarrier);
                Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
                for (size_t iSymb = 0; iSymb < nSymb; iSymb++) {
                    auto nSymNum = request->pSymbIndex[iSymb];
                    // convert rx pointer
                    for (size_t i = 0; i < N_RX; i++) {
                        pRxIn[i] = reinterpret_cast<T *>
                            (reinterpret_cast<int32_t *>(request->pRxSignal[i][nSymNum])
                            + nStartSC + nSCIdx);
                        rxIn[i] = loadu(pRxIn[i]);
                        _mm_prefetch(pRxIn[i] + 1, _MM_HINT_T2);
                    }

                    // If it is needed to update interpolated CE, otherwise previous sym parameters reused
                    if (flag_symH_upd[nSymNum] == 1) {
                        // #pragma unroll(N_TX)
                        for (size_t j = 0; j < N_TX; j ++) {
                            #pragma unroll(N_RX)
                            for (size_t i = 0; i < N_RX; i++) {
                                auto pChBase = reinterpret_cast<int32_t *>(request->pChState[i][j]) + nStartSC + nSCIdx;
                                auto pChIn0 = reinterpret_cast<T *>(pChBase);
                                auto pChIn1 = reinterpret_cast<T *>(pChBase + offsetDMRS);
                                T temp0 = loadu(pChIn0);
                                T temp1 = loadu(pChIn1);
                                temp0 = temp0 * wSymAvx[nSymNum][0]; // _mm512_mul_ph(_mm512_set1_ph(lhs), rhs.vec)
                                temp1 = temp1 * wSymAvx[nSymNum][1]; // _mm512_mul_ph(_mm512_set1_ph(lhs), rhs.vec)
                                ChIn[j][i] = temp0 + temp1; // _mm512_add_ph
                            }
                        }
                        //1. A = H' * H + Sigma2
                        HxH<T, N_TX, N_RX> ( ftempARe, ftempBRe, ChIn, avxfSigma2);

                        // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                        // matrix_print(ftempBRe);
                        matrix_inverse<T, N_TX>(ftempBRe);
                        // matrix_print(finvARe);

                        #pragma unroll(N_TX)
                        for (size_t i = 0; i < N_TX; i++) {
                            // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
                            auto gain = postSINRCalc (i, ftempBRe, ftempARe);
                            // temp used
                            ftempPostSINR[i] = duplicateReal(gain);
                        }
                    }

                    #pragma unroll(N_TX)
                    for (size_t i = 0; i < N_TX; i++) {
                        auto pPostSinr = reinterpret_cast<T *>(reinterpret_cast<float *>(response->pPostSINR[i])
                                        + nSymNum * nAlignedTotalSubCarrier + nStartSC + nSCIdx);
                        storeu(pPostSinr, kMask, ftempPostSINR[i]);
                    }
                    // 3. Z = H' * y
                    #pragma unroll(N_TX)
                    for (size_t i = 0; i < N_TX; i++) {
                        // calculate the real part of z
                        // calculate the imag part of z
                        // ftempZRe[i] = acc_sum<T, N_RX>(ChIn[i], pRxIn);
                        ftempZRe[i] = dotC<T, N_RX>(rxIn, ChIn[i]);
                        // ftempZRe[i] = dotC<T, N_RX>(ChIn[i], rxIn);
                    }

                    // 4. x = invA * z
                    #pragma unroll(N_TX)
                    for(size_t i = 0; i < N_TX; i ++) {
                        // real imag part
                        auto tx = dot<T, N_TX>(ftempBRe[i], ftempZRe);
                        tx = tx * avxShift;
                        // store x
                                            // store x
                        auto pTx = reinterpret_cast<T *>(
                                    reinterpret_cast<int32_t *>(response->pEstTxSignal[i][nSymNum])
                                    + nStartSC + nSCIdx);
                        storeu(pTx, kMask, tx);
                        _mm_prefetch(pTx + 1, _MM_HINT_T2);
                    }
                }//end the Symbol cycle
            }
        }
    }
#endif

    static void mimo_mmse_avx512_lin_interp(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response, Is16vec32 d) {
        using T = Is16vec32;
        using FloatType = typename DataType<T>::FloatType;
        using Float = typename DataType<T>::Float;
        T ChIn[N_TX][N_RX];
        T * pRxIn[N_RX];
        T offsetExp[BBLIB_N_SYMB_PER_SF][N_TX];
        T wSymAvx[BBLIB_N_SYMB_PER_SF][2];

        FloatType ftempARe[N_TX][N_TX];//float A = H'*H + sigam2
        FloatType ftempZRe[N_TX];//float Z=H'*Y
        FloatType ftempBRe[N_TX][N_TX];
        F32vec16 ftempPostSINR[N_TX];

        T ChImNegRe[N_TX][N_RX];
        FloatType ftempAIm[N_TX][N_TX];//float A = H'*H + sigam2
        FloatType finvAIm[N_TX][N_TX];//float invA
        FloatType finvARe[N_TX][N_TX];//float invA
        FloatType ftempZIm[N_TX];//float Z=H'*Y
        FloatType ftempBIm[N_TX][N_TX];

        const auto mmse_x_left = (Float)(1 << BBLIB_MMSE_X_LEFT_SHIFT);

        //sigma
        const Float sigma2 = (Float)(request->nSigma2);
        const auto left_shift = (N_TX == 4 ? 1.0 / mmse_x_left : mmse_x_left);
        const auto avxShift = FloatType(left_shift);
        const auto avxfSigma2 = FloatType(sigma2);
        const auto nFactor = 1.0 / (Float)(1 << BBLIB_MMSE_LEMMA_SCALING);
        const auto avxGainShift = (N_TX == 4 ? FloatType(nFactor) : FloatType(1.0));

        int16_t refSym = 0;
        int32_t compOffset;

        int16_t nFftSize = request->nFftSize;
        int16_t nMaxCp = 0, nMinCp = 0;
        int16_t nNumerology = request->nNumerology;
        int16_t *pFoCompScCp = request->pFoCompScCp;

        if (request->nEnableFoComp)
        {
            if (unlikely(-1 == bblib_get_sys_params(nNumerology, nFftSize, &nMaxCp, &nMinCp, NULL)))
            {
                printf("Error! Not support this case nMu %d nFftSize %d\n", nNumerology, nFftSize);
                return;
            }
        }
        const int16_t nSubCarrier = request->nSubCarrier;
        const int16_t nAlignedTotalSubCarrier = request->nTotalAlignedSubCarrier;
        const int16_t nStartSC = request->nStartSC;
        const int16_t nSymb = request->nSymb;
        const int16_t nTime = nSubCarrier / 16;
        const int16_t nRestLen = nSubCarrier - nTime * 16;
        int16_t flag_symH_upd[BBLIB_N_SYMB_PER_SF] = { 0 };
        const size_t offsetDMRS =  N_RX * N_TX * nAlignedTotalSubCarrier;
        int16_t nDmrsChSymb = request->nDmrsChSymb;
        if (nDmrsChSymb != 2) {
            printf("Wrong API: only two DMRS symbols are supported currently with linear interpolation\n");
            return;
        } else {
            int16_t nMappingType = request->nMappingType;
            int16_t nDataSymb = request->nSymb;
            int16_t nGranularity = request->nGranularity;
            for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                auto nSymNum = request->pSymbIndex[iSymb];
                auto wSym0 = wType[nMappingType][nGranularity][nSymNum][0];
                auto wSym1 = wType[nMappingType][nGranularity][nSymNum][1];

                wSymAvx[nSymNum][0] = _mm512_set1_epi16(wSym0);
                wSymAvx[nSymNum][1] = _mm512_set1_epi16(wSym1);
            }
            for (size_t i = 0; i < BBLIB_N_SYMB_PER_SF; i++) {
                flag_symH_upd[i] = g_flag_symH_upd[nMappingType][nGranularity][i];
            }

            if (request->nEnableFoComp)
            {
                // auto a = foParaMap[nNumerology][nFftSize];
                // pFoCompScCp = std::get<0>(a);
                // nMaxCp = std::get<1>(a);
                // nMinCp = std::get<2>(a);

                // Compute FO compensation for each symbol
                for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++)
                {
                    auto nSymNum = request->pSymbIndex[iSymb];
                    if (flag_symH_upd[nSymNum] == 1) {
                        refSym = nSymNum;
                        compOffset = nSymNum * (nFftSize + nMinCp) + nMaxCp;
                    } else {
                        compOffset = (refSym - nSymNum)*(nFftSize + nMinCp);
                    }
                    for (size_t j = 0; j < N_TX; j++) {
                        int16_t compOffsetTemp = floor(compOffset * request->fEstCfo[j] * nFftSize);
                        compOffsetTemp = (compOffsetTemp + nFftSize) % nFftSize;
                        // auto pTempX = reinterpret_cast<T *>(&pFoCompScCp[compOffsetTemp * 2 * 16]);
                        // offsetExp[nSymNum][j] = loadu(pTempX);
                        auto pTempX = reinterpret_cast<int32_t *>(&pFoCompScCp[compOffsetTemp * 2]);
                        offsetExp[nSymNum][j] = _mm512_set1_epi32(*pTempX);
                    }
                }
            }

            // loop channel symbol
            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {
                // Mask16 kMask = (nSCIdx + 16 > nSubCarrier) ?
                //         (static_cast<Mask16>(1) << nRestLen) - 1 : 0xffff;
                Mask16 scFlag = (nSCIdx + 16 > nSubCarrier);
                Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
                for (size_t iSymb = 0; iSymb < nSymb; iSymb++) {
                    auto nSymNum = request->pSymbIndex[iSymb];
                    // convert rx pointer
                    for (size_t i = 0; i < N_RX; i++) {
                        pRxIn[i] = reinterpret_cast<T *>
                            (reinterpret_cast<int32_t *>(request->pRxSignal[i][nSymNum])
                            + nStartSC + nSCIdx);
                        _mm_prefetch(pRxIn[i] + 1, _MM_HINT_T2);
                    }

                    // If it is needed to update interpolated CE, otherwise previous sym parameters reused
                    if (flag_symH_upd[nSymNum] == 1) {
                        // #pragma unroll(N_TX)
                        for (size_t j = 0; j < N_TX; j ++) {
                            #pragma unroll(N_RX)
                            for (size_t i = 0; i < N_RX; i++) {
                                auto pChBase = reinterpret_cast<int32_t *>(request->pChState[i][j]) + nStartSC + nSCIdx;
                                auto pChIn0 = reinterpret_cast<T *>(pChBase);
                                auto pChIn1 = reinterpret_cast<T *>(pChBase + offsetDMRS);
                                T temp0 = loadu(pChIn0);
                                T temp1 = loadu(pChIn1);
                                temp0 = mulhrs(temp0, wSymAvx[nSymNum][0]);
                                temp1 = mulhrs(temp1, wSymAvx[nSymNum][1]);
                                ChIn[j][i] = (temp0 + temp1) << 1;
                                IM_NEG_RE(ChIn[j][i], ChImNegRe[j][i]);
                            }
                        }
                        //1. A = H' * H + Sigma2
                        //calculate the real part of H' * H
                        // #pragma unroll(N_TX)
                        for (size_t i = 0; i < N_TX; i ++) {
                            ftempARe[i][i] = acc_sum<N_RX>(ChIn[i], ChIn[i]);
                            ftempAIm[i][i] = FloatType(0.0);
                            // ftempAIm[i][i] = allfZero;
                            // B = A + sigma2
                            ftempBRe[i][i] = ftempARe[i][i] + avxfSigma2;
                            ftempBIm[i][i] = FloatType(0.0);
                        }

                        HxHReal<N_TX, N_RX> ( ftempARe, ftempBRe, ChIn);

                        // calculate the imag part of H' * H
                        HxHImage<N_TX, N_RX> (ftempAIm, ftempBIm, ChImNegRe, ChIn);

                        // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                        matrix_inverse<N_TX>(ftempBRe, ftempBIm, finvARe, finvAIm);

                        #pragma unroll(N_TX)
                        for (size_t i = 0; i < N_TX; i++) {
                            // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
                            ftempPostSINR[i] = postSINRCalc (i, avxGainShift, finvAIm, finvARe, ftempAIm, ftempARe);
                        }
                    }
                    #pragma unroll(N_TX)
                    for (size_t i = 0; i < N_TX; i++) {
                        auto pPostSinr = reinterpret_cast<float *>(response->pPostSINR[i])
                                        + nSymNum * nAlignedTotalSubCarrier + nStartSC + nSCIdx;
                        storeu(pPostSinr, kMask, ftempPostSINR[i]);
                    }
                    // 3. Z = H' * y
                    #pragma unroll(N_TX)
                    for (size_t i = 0; i < N_TX; i++) {
                    // calculate the real part of z
                        ftempZRe[i] = acc_sum<N_RX>(ChIn[i], pRxIn);
                    // calculate the imag part of z
                        ftempZIm[i] = acc_sum<N_RX>(ChImNegRe[i], pRxIn);
                    }

                    // 4. x = invA * z
                    #pragma unroll(N_TX)
                    for(size_t i = 0; i < N_TX; i ++) {
                        auto tx = txCalc<N_TX>(i, avxShift, finvAIm, finvARe, ftempZRe, ftempZIm);
                        // store x
                        auto pTx = reinterpret_cast<T *>(
                                    reinterpret_cast<int32_t *>(response->pEstTxSignal[i][nSymNum])
                                    + nStartSC + nSCIdx);
                        storeu(pTx, kMask, tx);
                        // _mm_prefetch(pTx + 1, _MM_HINT_T2);
                    }
                }//end the Symbol cycle
            }
        }
    }
};
#undef IM_NEG_RE

template<size_t N_RX = 16, size_t N_TX = 16, typename T = Is16vec32>
void mimo_mmse_avx512(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response)
{
    MimoMmse<N_RX, N_TX>::mimo_mmse_avx512(request, response, T());
}

template<size_t N_RX = 16, size_t N_TX = 16, typename T = Is16vec32>
void mimo_mmse_avx512_interp(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response)
{
    if ((request->nLinInterpEnable == 1) &&
        (request->nDmrsChSymb != 1)) {
        MimoMmse<N_RX, N_TX>::mimo_mmse_avx512_lin_interp(request, response, T());
    } else {
        MimoMmse<N_RX, N_TX>::mimo_mmse_avx512(request, response, T());
    }
    return;
}
/*! \brief MMSE MIMO detection, with post SNR calculation. With AVX512 intrinsics.
    \param [in] request Input request structure for MMSE MIMO.
    \param [out] response Output response structure for MMSE MIMO..
    \return 0 for success, and -1 for error
*/
template<typename T = Is16vec32>
int32_t bblib_mimo_mmse_detection(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response)
{
    int32_t nLayer = request->nLayer;
    int32_t nRxAnt = request->nRxAnt;
    int32_t n_return = 0;
#if 0
    if (1 == nLayer) {
        if (1 == nRxAnt) {
            mimo_mmse_avx512<1, 1, T>(request, response);
        } else if (2 == nRxAnt) {
            mimo_mmse_avx512_interp<2, 1, T>(request, response);
        } else if (4 == nRxAnt) {
            mimo_mmse_avx512<4, 1, T>(request, response);
        } else if (8 == nRxAnt) {
            mimo_mmse_avx512<8, 1, T>(request, response);
        } else if (16 == nRxAnt) {
            mimo_mmse_avx512_interp<16, 1, T>(request, response);
        } else {
            printf("Error! Current MMSE doesn't support this combination! nLayer = %d, nRxAnt = %d\n", nLayer, nRxAnt);
            n_return = -1;
        }
    } else if (2 == nLayer) {
        if (2 == nRxAnt) {
            mimo_mmse_avx512_interp<2, 2, T>(request, response);
        } else if (4 == nRxAnt) {
            mimo_mmse_avx512_interp<4, 2, T>(request, response);
        } else if (8 == nRxAnt) {
            mimo_mmse_avx512<8, 2, T>(request, response);
        } else if (16 == nRxAnt) {
            mimo_mmse_avx512_interp<16, 2, T>(request, response);
        } else {
            printf("Error! Current MMSE doesn't support this combination! nLayer = %d, nRxAnt = %d\n", nLayer, nRxAnt);
            n_return = -1;
        }
    } else if (4 == nLayer) {
        if (4 == nRxAnt) {
            mimo_mmse_avx512_interp<4, 4, T>(request, response);
        } else if (8 == nRxAnt) {
            mimo_mmse_avx512<8, 4, T>(request, response);
        } else if (16 == nRxAnt) {
            mimo_mmse_avx512_interp<16, 4, T>(request, response);
        } else {
            printf("Error! Current MMSE doesn't support this combination! nLayer = %d, nRxAnt = %d\n", nLayer, nRxAnt);
            n_return = -1;
        }
    } else if (8 == nLayer) {
        if (8 == nRxAnt) {
            mimo_mmse_avx512<8, 8, T>(request, response);
        } else if (16 == nRxAnt) {
            mimo_mmse_avx512_interp<16, 8, T>(request, response);
        } else {
            printf("Error! Current MMSE doesn't support this combination! nLayer = %d, nRxAnt = %d\n", nLayer, nRxAnt);
            n_return = -1;
        }
    } else if (16 == nLayer) {
        if (16 == nRxAnt) {
            mimo_mmse_avx512<16, 16, T>(request, response);
        } else {
            printf("Error! Current MMSE doesn't support this combination! nLayer = %d, nRxAnt = %d\n", nLayer, nRxAnt);
            n_return = -1;
        }
    } else {
        printf("Error! Current MMSE doesn't support this combination! nLayer = %d, nRxAnt = %d\n", nLayer, nRxAnt);
        n_return = -1;
    }
    return (n_return);
#endif
}

/*! \brief MMSE MIMO detection, with post SNR calculation. With AVX512 intrinsics.
    \param [in] request Input request structure for MMSE MIMO.
    \param [out] response Output response structure for MMSE MIMO..
    \return 0 for success, and -1 for error
*/
int32_t bblib_mimo_mmse_detection_avx512(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response) {
    return bblib_mimo_mmse_detection<Is16vec32>(request, response);

}
#ifdef _BBLIB_SPR_
int32_t bblib_mimo_mmse_detection_avx512_5gisa(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response) {
    return bblib_mimo_mmse_detection<CF16vec16>(request, response);
}
#endif
#endif
