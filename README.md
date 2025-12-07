# videoFilterGraySobel

## About

This repository is a collaboration between Yexall and I to create an efficient Sobel filtering program in C++ that runs on a Raspberry Pi for our embedded systems class.
The program allows the user to filter videos with a grayscale, or implement a Sobel filter on top of the image as a video runs.

## Filter Types

The following filter types are supported:

- Plain: No filtering is performed, but each frame is processed
- Gray: Black, gray, and white coloration
- Sobel: Convert the grayscale-filtered image to a Sobel filter

### Grayscale

Each frame of the video is filtered using the BT.709 grayscale standard with the given formula:
$$ Y = 0.2126R + 0.7152G + 0.0722B $$
This formula highlights how each pixel value must be multiplied by a given value to convert it to a proper grayscale equivalent.
Given that pixel color values in RGB format are 8-bit values (0-255), we can simply scale these values to equivalent values before multiplying and bitshifting by 8 bits to simplify the math.

### Sobel

A Sobel filter is used in computer vision and image processing to detect edges in an image. For any grayscale image,convolving a grayscale image along the horizontal and vertical axes using the following two matrices, $G_x$ and $G_y$, creates the Sobel filter version of the image.

$$G_x=\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix} * A$$

$$G_y=\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
-1 & 2 & 1
\end{bmatrix}*A$$

This transforms the previously completed grayscale image to only display the edges of the objects in that image, turning everything inside the object black.

## Prerequisites

This program requires the user to have the OpenCV library and the GCC compiler installed to compile and run.
For a Debian-based operating system (i.e. Raspbian or Ubuntu), this can be done as shown:

``` {bash}
sudo apt update -y
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt install git build-essential libopencv-dev -y
```

## Directions

Compile the program with the following command:
`make`

To run this program, use the following command:
`./filter <VIDEO_FILE> <FILTER_TYPE>`

The only valid types are, once again:
- plain
- gray
- sobel
