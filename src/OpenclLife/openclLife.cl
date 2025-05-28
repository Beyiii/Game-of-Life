typedef unsigned char uchar;

__kernel void game_of_life(__global const uchar* data,
                           __global uchar* result,
                           const uint width,
                           const uint height) {

    int idx = get_global_id(0);
    int total = width * height;
    if (idx >= total) return;

    int x = idx % width;
    int y = idx / width;

    // Coordenadas con wrap-around
    int x0 = (x + width - 1) % width;
    int x2 = (x + 1) % width;
    int y0 = (y + height - 1) % height;
    int y2 = (y + 1) % height;

    // Acceder por coordenadas
    #define AT(xx,yy) data[(yy) * width + (xx)]

    uchar alive =
        AT(x0, y0) + AT(x, y0) + AT(x2, y0) +
        AT(x0, y)             + AT(x2, y) +
        AT(x0, y2) + AT(x, y2) + AT(x2, y2);

    uchar center = AT(x, y);
    result[idx] = (alive == 3 || (alive == 2 && center)) ? 1 : 0;
}
